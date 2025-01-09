#include "lstm_model.h"
#include <random>

void LSTM_Model::load_model (const nlohmann::json& model_json)
{
    model_json_original = model_json;
    load_model (model_json_original["state_dict"],
                model_json_original["model_data"]["hidden_size"].get<int>());
}

template <typename Fn, size_t... Ix>
constexpr void for_each_index (Fn&& fn, std::index_sequence<Ix...>) noexcept
{
    (void) std::initializer_list<int> { ((void) fn (std::integral_constant<std::size_t, Ix>()), 0)... };
}

/** utils for making offset index sequences */
template <std::size_t N, typename Seq>
struct offset_sequence;

template <std::size_t N, std::size_t... Ints>
struct offset_sequence<N, std::index_sequence<Ints...>>
{
    using type = std::index_sequence<Ints + N...>;
};
template <std::size_t N, typename Seq>
using offset_sequence_t = typename offset_sequence<N, Seq>::type;

template <std::size_t start, std::size_t end_inclusive>
using range_sequence = offset_sequence_t<start, std::make_index_sequence<end_inclusive - start + 1>>;

void LSTM_Model::load_model (const nlohmann::json& state_dict, int hidden_size)
{
    juce::SpinLock::ScopedLockType model_loading_lock { model_loading_mutex };

    for_each_index (
        [this, hidden_size] (auto i)
        {
            if (i == hidden_size)
            {
                std::cout << "Loading model with hidden size: " << hidden_size << '\n';
                model_variant.emplace<Model<i>>();
            }
        },
        range_sequence<min_hidden_size, max_hidden_size> {});

    std::visit (
        [&state_dict] (auto& model)
        {
            RTNeural::torch_helpers::loadLSTM<float> (state_dict, "rec.", model.template get<0>());
            RTNeural::torch_helpers::loadDense<float> (state_dict, "lin.", model.template get<1>());
        },
        model_variant);
}

void LSTM_Model::process (std::span<float> data, float param)
{
    juce::SpinLock::ScopedTryLockType model_loading_lock { model_loading_mutex };
    if (! model_loading_lock.isLocked())
        return;

    std::visit (
        [data, param] (auto& model)
        {
            alignas (16) float input[4] { 0.0f, param };
            for (auto& x : data)
            {
                input[0] = x;
                x += model.forward (input);
            }
        },
        model_variant);
}

static auto get_model (int hidden_size)
{
    auto model = std::make_unique<RTNeural::Model<float>> (LSTM_Model::input_size);
    model->addLayer (new RTNeural::LSTMLayer<float> { LSTM_Model::input_size, hidden_size });
    model->addLayer (new RTNeural::Dense<float> { hidden_size, 1 });
    return model;
}

static auto get_test_data()
{
    std::mt19937 gen { 0x1234 }; // Fixed random seed so we get the same data every time
    std::uniform_real_distribution<float> dist { -1.0f, 1.0f };
    std::vector<float> data {};
    data.resize (8192);
    for (int n = 0; n < data.size(); ++n)
        data[n] = dist (gen);
    return data;
}

static auto run_model (const nlohmann::json& state_dict, int hidden_size)
{
    auto model = get_model (hidden_size);
    // Model_Type model {};
    RTNeural::torch_helpers::loadLSTM<float> (state_dict, "rec.", *dynamic_cast<RTNeural::LSTMLayer<float>*> (model->layers[0]));
    RTNeural::torch_helpers::loadDense<float> (state_dict, "lin.", *dynamic_cast<RTNeural::Dense<float>*> (model->layers[1]));

    auto data = get_test_data();
    alignas (16) float input[4] {};
    for (auto& x : data)
    {
        input[0] = x;
        x = model->forward (input);
    }

    return data;
}

static auto compute_rms_error (std::span<const float> x, std::span<const float> y)
{
    auto square_error_accum = 0.0f;
    for (int n = 0; n < x.size(); ++n)
    {
        const auto sample_error = x[n] - y[n];
        square_error_accum += sample_error * sample_error;
    }

    return std::sqrt (square_error_accum / static_cast<float> (x.size()));
}

static auto test_channel_prune (nlohmann::json state_dict_prune,
                                std::span<const float> ground_truth_output,
                                int channel_to_prune,
                                int hidden_size,
                                bool verbose = false)
{
    if (verbose)
        std::cout << "Pruning dense channel " << channel_to_prune << "... ";

    state_dict_prune["lin.weight"][0][channel_to_prune] = 0.0f;
    for (int i = 3; i >= 0; i--)
    {
        for (auto& value : state_dict_prune["rec.weight_ih_l0"][channel_to_prune + hidden_size * i])
            value = 0.0f;
        for (auto& value : state_dict_prune["rec.weight_hh_l0"][channel_to_prune + hidden_size * i])
            value = 0.0f;

        state_dict_prune["rec.bias_ih_l0"][channel_to_prune + hidden_size * i] = 0.0f;
        state_dict_prune["rec.bias_hh_l0"][channel_to_prune + hidden_size * i] = 0.0f;
    }
    for (auto& row : state_dict_prune["rec.weight_hh_l0"])
        row[channel_to_prune] = 0.0f;

    auto test_output = run_model (state_dict_prune, hidden_size);

    const auto rms_error = compute_rms_error (test_output, ground_truth_output);
    if (verbose)
        std::cout << "RMS Error: " << rms_error << '\n';
    return rms_error;
}

static auto model_prune (nlohmann::json& state_dict,
                         std::span<const float> ground_truth_output,
                         int hidden_size)
{
    std::vector<float> error_vals (hidden_size);
    for (int i = 0; i < hidden_size; ++i)
        error_vals[i] = test_channel_prune (state_dict, ground_truth_output, i, hidden_size);

    const auto min_iter = std::min_element (error_vals.begin(), error_vals.end());
    const auto min_channel = std::distance (error_vals.begin(), min_iter);
    std::cout << "Best channel to prune: " << min_channel << ", RMS Error: " << *min_iter << '\n';
    return std::make_tuple (min_channel, *min_iter);
}

static auto prune_channel (nlohmann::json& state_dict, int prune_channel, int& hidden_size, std::span<const float> ground_truth_output)
{
    std::cout << "Pruning channel: " << prune_channel << "... ";
    state_dict["lin.weight"][0].erase (prune_channel);
    for (int i = 3; i >= 0; i--)
    {
        state_dict["rec.weight_ih_l0"].erase (prune_channel + hidden_size * i);
        state_dict["rec.weight_hh_l0"].erase (prune_channel + hidden_size * i);

        state_dict["rec.bias_ih_l0"].erase (prune_channel + hidden_size * i);
        state_dict["rec.bias_hh_l0"].erase (prune_channel + hidden_size * i);
    }
    for (auto& row : state_dict["rec.weight_hh_l0"])
        row.erase (prune_channel);

    hidden_size--;
    const auto prune_output = run_model (state_dict, hidden_size);
    const auto prune_rms_error = compute_rms_error (ground_truth_output, prune_output);
    std::cout << "RMS Error: " << prune_rms_error << "\n\n";

    return prune_output;
}

static auto prune_model (nlohmann::json state_dict, int hidden_size)
{
    auto ground_truth_output = run_model (state_dict, hidden_size);

    static constexpr auto hidden_size_threshold = 4;
    static constexpr auto pruning_error_threshold = 0.05;
    std::cout << "Pruning error threshold: " << pruning_error_threshold << '\n';
    std::cout << "Pruning hidden size threshold: " << hidden_size_threshold << '\n';
    while (hidden_size > hidden_size_threshold)
    {
        const auto [channel_to_prune, prune_error] = model_prune (state_dict, ground_truth_output, hidden_size);
        if (prune_error > pruning_error_threshold)
        {
            std::cout << "Pruning error greater than threshold...\n\n";
            break;
        }
        ground_truth_output = prune_channel (state_dict, channel_to_prune, hidden_size, ground_truth_output);
    }

    return std::make_pair (state_dict, hidden_size);
}

void LSTM_Model::prune_model()
{
    const auto [pruned_state_dict, pruned_hidden_size] = ::prune_model (model_json_original["state_dict"],
                                                                        model_json_original["model_data"]["hidden_size"].get<int>());
    load_model (pruned_state_dict, pruned_hidden_size);
}
