#include <iostream>
#include <random>

#include <RTNeural/RTNeural.h>

static constexpr int N = 8192;
static constexpr int input_size = 2;
// static constexpr int hidden_size = 24;
// using Model = RTNeural::ModelT<float,
//                                input_size,
//                                1,
//                                RTNeural::LSTMLayerT<float, input_size, hidden_size>,
//                                RTNeural::DenseT<float, hidden_size, 1>>;

// using Model_Prune1 = RTNeural::ModelT<float,
//                                       input_size,
//                                       1,
//                                       RTNeural::LSTMLayerT<float, input_size, hidden_size - 1>,
//                                       RTNeural::DenseT<float, hidden_size - 1, 1>>;

auto get_model (int hidden_size)
{
    auto model = std::make_unique<RTNeural::Model<float>> (input_size);
    model->addLayer (new RTNeural::LSTMLayer<float> { input_size, hidden_size });
    model->addLayer (new RTNeural::Dense<float> { hidden_size, 1 });
    return model;
}

auto get_test_data()
{
    std::mt19937 gen { 0x1234 }; // Fixed random seed so we get the same data every time
    std::uniform_real_distribution<float> dist { -1.0f, 1.0f };
    std::vector<float> data {};
    data.resize (N);
    for (int n = 0; n < N; ++n)
        data[n] = dist (gen);
    return data;
}

auto run_model (const nlohmann::json& state_dict, int hidden_size)
{
    auto model = get_model (hidden_size);
    // Model_Type model {};
    RTNeural::torch_helpers::loadLSTM<float> (state_dict, "rec.", *dynamic_cast<RTNeural::LSTMLayer<float>*> (model->layers[0]));
    RTNeural::torch_helpers::loadDense<float> (state_dict, "lin.", *dynamic_cast<RTNeural::Dense<float>*> (model->layers[1]));

    auto data = get_test_data();
    alignas(16) float input[4] {};
    for (auto& x : data)
    {
        input[0] = x;
        x = model->forward (input);
    }

    return data;
}

auto compute_rms_error (std::span<const float> x, std::span<const float> y)
{
    auto square_error_accum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        const auto sample_error = x[n] - y[n];
        square_error_accum += sample_error * sample_error;
    }

    return std::sqrt (square_error_accum / static_cast<float> (N));
}

auto test_channel_prune (nlohmann::json state_dict_prune,
                         std::span<const float> ground_truth_output,
                         int channel_to_prune,
                         int hidden_size,
                         bool verbose = false)
{
    if (verbose) std::cout << "Pruning dense channel " << channel_to_prune << "... ";

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
    if (verbose) std::cout << "RMS Error: " << rms_error << '\n';
    return rms_error;
}

auto model_prune (nlohmann::json& state_dict,
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

auto prune_channel (nlohmann::json& state_dict, int prune_channel, int& hidden_size, std::span<const float> ground_truth_output)
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

int main()
{
    std::cout << "LSTM pruning test\n";

    const auto model_path { std::string { MODELS_DIR } + "/fuzz_2.json" };
    std::cout << "Testing model: " << model_path << std::endl;
    nlohmann::json model_json {};
    std::ifstream { model_path, std::ifstream::binary } >> model_json;
    assert(model_json["model_data"]["input_size"].get<int>() == input_size);
    assert(model_json["model_data"]["output_size"].get<int>() == 1);
    assert(model_json["model_data"]["unit_type"].get<std::string_view>() == "LSTM");

    auto hidden_size = model_json["model_data"]["hidden_size"].get<int>();
    auto& state_dict = model_json.at ("state_dict");
    auto ground_truth_output = run_model (state_dict, hidden_size);

    static constexpr auto pruning_error_threshold = 0.05;
    std::cout << "Pruning error threshold: " << pruning_error_threshold << '\n';
    while (hidden_size > 4)
    {
        const auto [channel_to_prune, prune_error] = model_prune (state_dict, ground_truth_output, hidden_size);
        if (prune_error > pruning_error_threshold)
        {
            std::cout << "Pruning error greater than threshold...\n\n";
            break;
        }
        ground_truth_output = prune_channel (state_dict, channel_to_prune, hidden_size, ground_truth_output);
    }

    std::cout << "Pruned hidden size: " << hidden_size << '\n';

    return 0;
}

// Pruning dense channel 0... RMS Error: 0.220418
// Pruning dense channel 1... RMS Error: 0.136846
// Pruning dense channel 2... RMS Error: 0.239895
// Pruning dense channel 3... RMS Error: 0.0857022
// Pruning dense channel 4... RMS Error: 9.82266e-07
// Pruning dense channel 5... RMS Error: 0.343273
// Pruning dense channel 6... RMS Error: 0.137796
// Pruning dense channel 7... RMS Error: 0.163197
// Pruning dense channel 8... RMS Error: 0.174898
// Pruning dense channel 9... RMS Error: 0.093074
// Pruning dense channel 10... RMS Error: 0.158675
// Pruning dense channel 11... RMS Error: 0.0803543
// Pruning dense channel 12... RMS Error: 0.0600532
// Pruning dense channel 13... RMS Error: 0.0505263
// Pruning dense channel 14... RMS Error: 0.0422307
// Pruning dense channel 15... RMS Error: 0.0313095
// Pruning dense channel 16... RMS Error: 0.055394
// Pruning dense channel 17... RMS Error: 0.149035
// Pruning dense channel 18... RMS Error: 0.14877
// Pruning dense channel 19... RMS Error: 0.141735
// Pruning dense channel 20... RMS Error: 0.0595955
// Pruning dense channel 21... RMS Error: 0.0272432
// Pruning dense channel 22... RMS Error: 0.217288
// Pruning dense channel 23... RMS Error: 0.0962931
