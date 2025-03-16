#include <future>
#include <iostream>
#include <random>
#include <sndfile.h>

#include <RTNeural/RTNeural.h>

static std::tuple<std::vector<float>, std::vector<float>> get_audio_data()
{
    // re-use the same data that we used for training
    static constexpr int seek_offset = 2'000'000;
    static constexpr int num_samples = 1'471'622;

    // Or, use validation data
    // static constexpr int seek_offset = 1'500'000;
    // static constexpr int num_samples = 500'000;

    std::vector<float> in_data (num_samples);
    std::vector<float> target_data (num_samples);

    {
        const auto audio_path { std::string { TRAIN_DIR } + "/fuzz_input.wav" };
        SF_INFO audio_file_info;
        auto* audio_file = sf_open (audio_path.c_str(), SFM_READ, &audio_file_info);
        sf_seek (audio_file, seek_offset, SEEK_SET);
        sf_read_float (audio_file, in_data.data(), (sf_count_t) in_data.size());
        sf_close (audio_file);
    }

    {
        // this file is stereo, but we're only going to use the left channel
        const auto audio_path { std::string { TRAIN_DIR } + "/fuzz_15_50.wav" };
        SF_INFO audio_file_info;
        auto* audio_file = sf_open (audio_path.c_str(), SFM_READ, &audio_file_info);
        sf_seek (audio_file, seek_offset, SEEK_SET);

        std::vector<float> stereo_data (num_samples * 2);
        sf_read_float (audio_file, stereo_data.data(), (sf_count_t) target_data.size());
        sf_close (audio_file);

        for (int n = 0; n < num_samples; ++n)
            target_data[n] = stereo_data[2 * n];
    }

    return std::make_tuple (in_data, target_data);
}

static auto get_model_json()
{
    const auto model_path { std::string { TRAIN_DIR } + "/lstm.json" };
    nlohmann::json model_json {};
    std::ifstream { model_path, std::ifstream::binary } >> model_json;
    return model_json;
}

static auto compute_mse (std::span<const float> x, std::span<const float> y)
{
    const auto N = x.size();

    auto square_error_accum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        const auto sample_error = (x[n] - y[n]);
        square_error_accum += sample_error * sample_error;
    }

    return square_error_accum / static_cast<float> (N);
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

struct Model
{
    static constexpr int input_size = 1;
    static constexpr int max_hidden_size = 84;
    static constexpr int min_hidden_size = 48;

    template <int hidden_size>
    struct LSTM_Model
    {
        RTNeural::LSTMLayerT<float, 1, hidden_size> lstm {};
        RTNeural::DenseT<float, hidden_size, 1> dense {};
    };

    template <typename T, typename... Args>
    struct concatenator;
    template <typename... Args0, typename... Args1>
    struct concatenator<std::variant<Args0...>, Args1...>
    {
        using type = std::variant<Args0..., Args1...>;
    };

    template <typename... Args0, typename Args1>
    struct concatenator<std::variant<Args0...>, Args1>
    {
        using type = std::variant<Args0..., Args1>;
    };

    template <int hidden_size>
    struct Model_Variant_Builder
    {
        using type = typename concatenator<typename Model_Variant_Builder<hidden_size - 1>::type, LSTM_Model<hidden_size>>::type;
    };

    template <>
    struct Model_Variant_Builder<min_hidden_size>
    {
        using type = std::variant<LSTM_Model<min_hidden_size>>;
    };
    using Model_Variant = Model_Variant_Builder<max_hidden_size>::type;

    Model_Variant model_variant {};
    int current_hidden_size {};

    Model (const nlohmann::json& model_json)
    {
        current_hidden_size = model_json["layers"][0]["shape"].back().get<int>();

        for_each_index (
            [this] (auto i)
            {
                if (i == current_hidden_size)
                {
                    // std::cout << "Loading model with hidden size " << current_hidden_size << std::endl;
                    model_variant.emplace<LSTM_Model<i>>();
                }
            },
            range_sequence<min_hidden_size, max_hidden_size> {});

        std::visit (
            [&model_json] (auto& model)
            {
                auto& lstm_weights = model_json["layers"][0]["weights"];
                auto& dense_weights = model_json["layers"][1]["weights"];
                RTNeural::json_parser::loadLSTM<float> (model.lstm, lstm_weights);
                RTNeural::json_parser::loadDense<float> (model.dense, dense_weights);
            },
            model_variant);
    }

    void process (std::span<float> data, float param);
};

static int count_params (const nlohmann::json& model_json)
{
    int count = 0;
    for (auto& layer : model_json["layers"])
    {
        for (auto& weights_matrix : layer["weights"])
        {
            if (weights_matrix[0].is_array())
            {
                for (auto& row : weights_matrix)
                    count += row.size();
            }
            else
            {
                count += weights_matrix.size();
            }
        }
    }
    return count;
}

static std::vector<float> run_model (Model& model, std::span<const float> input, bool verbose = true)
{
    std::vector<float> out (input.size());

    const auto start = std::chrono::high_resolution_clock::now();

    std::visit (
        [input, &out] (auto& model)
        {
            for (size_t n = 0; n < input.size(); ++n)
            {
                Eigen::Matrix<float, 1, 1> in { input[n] };
                model.lstm.forward (in);
                model.dense.forward (model.lstm.outs);
                out[n] = model.dense.outs (0);
            }
        },
        model.model_variant);

    const auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto test_duration_seconds = std::chrono::duration<float> { duration }.count();
    if (verbose)
        std::cout << "Inference Time: " << test_duration_seconds << " seconds" << std::endl;

    return out;
}

struct Pruning_Candidate
{
    int idx {};
    float value { 0.0f };
};

static nlohmann::json prune (nlohmann::json model_json,
                             std::span<Pruning_Candidate> candidates_to_prune)
{
    std::cout << "Pruning " << candidates_to_prune.size() << " structural elements...\n";

    for (int prune_idx = 0; prune_idx < candidates_to_prune.size(); ++prune_idx)
    {
        const auto& to_prune = candidates_to_prune[prune_idx];
        const auto hidden_size = model_json["layers"][0]["shape"].back().get<int>();
        auto& lstm_weights = model_json["layers"][0]["weights"];
        auto& kernel_weights = lstm_weights.at (0);
        auto& recurrent_weights = lstm_weights.at (1);
        auto& biases = lstm_weights.at (2);
        auto& dense_weights = model_json["layers"][1]["weights"][0];

        kernel_weights[0].erase (to_prune.idx + 3 * hidden_size);
        kernel_weights[0].erase (to_prune.idx + 2 * hidden_size);
        kernel_weights[0].erase (to_prune.idx + 1 * hidden_size);
        kernel_weights[0].erase (to_prune.idx + 0 * hidden_size);

        for (int i = 0; i < hidden_size; ++i)
        {
            recurrent_weights[i].erase (to_prune.idx + 3 * hidden_size);
            recurrent_weights[i].erase (to_prune.idx + 2 * hidden_size);
            recurrent_weights[i].erase (to_prune.idx + 1 * hidden_size);
            recurrent_weights[i].erase (to_prune.idx + 0 * hidden_size);
        }
        recurrent_weights.erase (to_prune.idx);

        biases.erase (to_prune.idx + 3 * hidden_size);
        biases.erase (to_prune.idx + 2 * hidden_size);
        biases.erase (to_prune.idx + 1 * hidden_size);
        biases.erase (to_prune.idx + 0 * hidden_size);

        dense_weights.erase (to_prune.idx);

        model_json["layers"][0]["shape"].back() = hidden_size - 1;

        for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
        {
            auto& to_fix = candidates_to_prune[fix_idx];
            if (to_fix.idx > to_prune.idx)
               to_fix.idx--;
        }

        // std::cout << kernel_weights.size() << '\n';
        // std::cout << kernel_weights[0].size() << '\n';
        // std::cout << recurrent_weights.size() << '\n';
        // std::cout << recurrent_weights[0].size() << '\n';
        // std::cout << biases.size() << '\n';
        // std::cout << dense_weights.size() << '\n';
        // std::cout << dense_weights[0].size() << '\n';
        // std::exit (0);
    }

    return model_json;
}

static float rank_min_weights (nlohmann::json model_json,
                               int idx)
{
    return 0.0f;
}

static float compute_mean (std::span<const float> values)
{
    return std::accumulate (values.begin(), values.end(), 0.0f) / static_cast<float> (values.size());
}

static float compute_stddev (std::span<const float> values, float mean)
{
    return std::sqrt (
        std::accumulate (values.begin(),
                         values.end(),
                         0.0f,
                         [mean] (float a, float b)
                         { return a + (b - mean) * (b - mean); })
        / static_cast<float> (values.size()));
}

static float rank_mean_activations (const nlohmann::json& model_json,
                                    int idx,
                                    std::span<const float> in_data)
{
    Model model { model_json };

    std::vector<float> activation_out (in_data.size());
    std::visit (
        [idx, in_data, &activation_out] (auto& model)
        {
            for (size_t n = 0; n < in_data.size(); ++n)
            {
                Eigen::Matrix<float, 1, 1> in { in_data[n] };
                model.lstm.forward (in);
                activation_out[n] = model.lstm.outs (idx);
            }
        },
        model.model_variant);

    const auto mean = compute_mean (activation_out);
    const auto variance = compute_stddev (activation_out, mean);
    return variance;
}

static float rank_minimization (nlohmann::json model_json,
                                int idx,
                                std::span<const float> in_data,
                                std::span<const float> target_data)
{
    Model model { model_json };

    std::vector<float> test_out (in_data.size());
    std::visit (
        [idx, in_data, &test_out] (auto& model)
        {
            for (size_t n = 0; n < in_data.size(); ++n)
            {
                Eigen::Matrix<float, 1, 1> in { in_data[n] };
                model.lstm.forward (in);
                model.lstm.outs (idx) = 0.0f;
                model.dense.forward (model.lstm.outs);
                test_out[n] = model.dense.outs (0);
            }
        },
        model.model_variant);

    const auto mse = compute_mse (test_out, target_data);
    return mse;
}

enum class Ranking
{
    Min_Weights,
    Mean_Activations,
    Minimization,
};

static auto rank_pruning_candidates (nlohmann::json model_json, Ranking ranking, std::span<const float> in_data, std::span<const float> target_data)
{
    const auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<void>> futures {};
    std::mutex mutex {};
    std::vector<Pruning_Candidate> candidates {};

    const auto hidden_size = model_json["layers"][0]["shape"].back().get<int>();

    for (int idx = 0; idx < hidden_size; ++idx)
    {
        futures.push_back (
            std::async (std::launch::async,
                        [ranking, model_json, idx, in_data, target_data, &candidates, &mutex]
                        {
                            float value {};
                            if (ranking == Ranking::Min_Weights)
                                value = rank_min_weights (model_json, idx);
                            else if (ranking == Ranking::Mean_Activations)
                                value = rank_mean_activations (model_json, idx, in_data);
                            else if (ranking == Ranking::Minimization)
                                value = rank_minimization (model_json, idx, in_data, target_data);

                            std::lock_guard lock { mutex };
                            candidates.push_back (Pruning_Candidate {
                                .idx = idx,
                                .value = value,
                            });
                        }));
    }

    for (auto& f : futures)
        f.wait();

    std::sort (candidates.begin(),
               candidates.end(),
               [] (const Pruning_Candidate& a, const Pruning_Candidate& b)
               {
                   return a.value < b.value;
               });

    const auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto test_duration_seconds = std::chrono::duration<float> { duration }.count();
    std::cout << "Ranking Time: " << test_duration_seconds << " seconds" << std::endl;

    return candidates;
}

int main()
{
    std::cout << "LSTM network pruning test\n";

    const auto [in_data, target_data] = get_audio_data();
    auto model_json = get_model_json();

    {
        std::cout << "Parameter count: " << count_params (model_json) << '\n';
        Model model { model_json };
        const auto model_out = run_model (model, in_data);
        std::cout << "Post-Training MSE: " << compute_mse (model_out, target_data) << '\n';
    }

    {
        // const auto ranking = Ranking::Min_Weights;
        const auto ranking = Ranking::Mean_Activations;
        // const auto ranking = Ranking::Minimization;
        auto pruning_candidates = rank_pruning_candidates (model_json, ranking, in_data, target_data);
        std::cout << "# Pruning Candidates: " << pruning_candidates.size() << '\n';

        static constexpr auto n_prune = 12;
        model_json = prune (model_json, std::span { pruning_candidates }.subspan (0, n_prune));
        std::cout << "Parameter count: " << count_params (model_json) << '\n';
        Model model { model_json };
        const auto model_out = run_model (model, in_data);
        std::cout << "Prune 1 MSE: " << compute_mse (model_out, target_data) << '\n';
    }

    return 0;
}
