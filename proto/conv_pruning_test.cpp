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
    const auto model_path { std::string { TRAIN_DIR } + "/conv.json" };
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

struct Model
{
    static constexpr auto num_layers = 4;
    static constexpr auto layer_width = 32;

    std::vector<RTNeural::Conv1D<float>> conv_layers {};
    RTNeural::TanhActivation<float> tanh_activation { layer_width };
    std::optional<RTNeural::Dense<float>> dense_layer {};
    alignas (16) std::array<std::array<float, layer_width>, num_layers + 1> layer_io {};

    explicit Model (const nlohmann::json& model_json)
    {
        conv_layers.reserve (num_layers);

        assert (model_json["in_shape"].back() == 1);
        int in_size = 1;
        for (auto& layer : model_json["layers"])
        {
            const auto out_size = layer["shape"].back().get<int>();
            // std::cout << layer["type"] << " {" << in_size << "," << out_size << "}\n";
            if (layer["type"] == "conv1d")
            {
                const auto kernel_size = layer["kernel_size"].back().get<int>();
                const auto dilation = layer["dilation"].back().get<int>();
                auto& conv1d = conv_layers.emplace_back (in_size,
                                                         out_size,
                                                         kernel_size,
                                                         dilation);
                RTNeural::json_parser::loadConv1D<float> (conv1d, kernel_size, dilation, layer["weights"]);
            }
            else if (layer["type"] == "dense")
            {
                auto& dense = dense_layer.emplace (in_size, out_size);
                RTNeural::json_parser::loadDense<float> (dense, layer["weights"]);
            }
            in_size = out_size;
        }
        assert (in_size == 1);
    }

    float forward (const float* in) noexcept
    {
        conv_layers.front().forward (in, layer_io.front().data());
        tanh_activation.forward (layer_io.front().data(), layer_io.front().data());

        for (int i = 0; i < num_layers - 1; ++i)
        {
            conv_layers[i + 1].forward (layer_io[i].data(), layer_io[i + 1].data());
            tanh_activation.forward (layer_io[i + 1].data(), layer_io[i + 1].data());
        }
        dense_layer->forward (layer_io[num_layers - 1].data(), layer_io[num_layers].data());

        return layer_io.back()[0];
    }
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
                {
                    if (row[0].is_array())
                    {
                        for (auto& el : row)
                            count += el.size();
                    }
                    else
                    {
                        count += row.size();
                    }
                }
            }
            else
            {
                count += weights_matrix.size();
            }
        }
    }
    return count;
}

template <typename Model_Type>
static std::vector<float> run_model (Model_Type& model, std::span<const float> input, bool verbose = true)
{
    std::vector<float> out (input.size());

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t n = 0; n < input.size(); ++n)
        out[n] = model.forward (&input[n]);

    const auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto test_duration_seconds = std::chrono::duration<float> { duration }.count();
    if (verbose)
        std::cout << "Inference Time: " << test_duration_seconds << " seconds" << std::endl;

    return out;
}

struct Pruning_Candidate
{
    int layer {};
    int row { -1 };
    float value { 0.0f };
};

static nlohmann::json prune (nlohmann::json model_json,
                             std::span<Pruning_Candidate> candidates_to_prune)
{
    std::cout << "Pruning " << candidates_to_prune.size() << " structural elements...\n";

    for (int prune_idx = 0; prune_idx < candidates_to_prune.size(); ++prune_idx)
    {
        const auto& to_prune = candidates_to_prune[prune_idx];

        auto& layers = model_json["layers"];
        for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
        {
            auto& layer = layers.at (layer_idx);
            if (layer["type"] != "conv1d")
                continue;

            // Convention:
            // Rows -> out size
            // Cols -> in size

            if (to_prune.layer == layer_idx)
            {
                auto& weights = layer["weights"].at (0);
                auto& biases = layer["weights"].at (1);

                for (auto& kernel : weights)
                {
                    for (auto& filter : kernel)
                        filter.erase (to_prune.row);
                }
                biases.erase (to_prune.row);
                layer["shape"].back() = layer["shape"].back().get<int>() - 1;

                auto next_layer_idx = layer_idx + 1;
                if (next_layer_idx < layers.size())
                {
                    auto& next_layer = layers.at (next_layer_idx);
                    auto& next_weights = next_layer["weights"].at (0);
                    if (next_layer["type"] == "conv1d")
                    {
                        for (auto& kernel : next_weights)
                            kernel.erase (to_prune.row);
                    }
                    else if (next_layer["type"] == "dense")
                    {
                        next_weights.erase (to_prune.row);
                    }
                }

                for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
                {
                    auto& to_fix = candidates_to_prune[fix_idx];
                    if (to_fix.layer == layer_idx && to_fix.row > to_prune.row)
                        to_fix.row--;
                }

                break;
            }
        }
    }

    return model_json;
}

static float rank_min_weights (const nlohmann::json& model_json,
                               int layer_idx,
                               int row)
{
    auto& layers = model_json["layers"].at (layer_idx);
    auto& weights = layers["weights"].at (0);

    float square_sum = 0.0f;
    for (auto& kernel : weights)
    {
        for (auto& filter : kernel)
        {
            auto v = filter.at (row).get<float>();
            square_sum += v * v;
        }
    }
    return square_sum;
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
                                    int layer_idx,
                                    int row,
                                    std::span<const float> in_data)
{
    Model model { model_json };

    std::vector<float> activation_out (in_data.size());
    for (size_t n = 0; n < in_data.size(); ++n)
    {
        model.conv_layers.front().forward (&in_data[n], model.layer_io.front().data());
        model.tanh_activation.forward (model.layer_io.front().data(), model.layer_io.front().data());

        for (int i = 0; i < layer_idx; ++i)
        {
            model.conv_layers[i + 1].forward (model.layer_io[i].data(), model.layer_io[i + 1].data());
            model.tanh_activation.forward (model.layer_io[i + 1].data(), model.layer_io[i + 1].data());
        }

        activation_out[n] = model.layer_io[layer_idx][row];
    }

    const auto mean = compute_mean (activation_out);
    const auto variance = compute_stddev (activation_out, mean);
    // std::cout << mean << " || " << variance << std::endl;

    return variance;
}

static float rank_minimization (nlohmann::json model_json,
                                int layer_idx,
                                int row,
                                std::span<const float> in_data,
                                std::span<const float> target_data)
{
    auto& layers = model_json["layers"].at (layer_idx);
    auto& weights = layers["weights"].at (0);

    for (auto& kernel : weights)
    {
        for (auto& filter : kernel)
        {
            filter.at (row) = 0.0f;
        }
    }

    Model model { model_json };
    const auto model_out = run_model (model, in_data, false);
    const auto mse = compute_mse (model_out, target_data);

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

    auto& layers = model_json["layers"];
    for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
    {
        auto& layer = layers.at (layer_idx);
        if (layer["type"] != "conv1d")
            continue;

        const auto rows = layer["shape"].back().get<int>();

        futures.push_back (
            std::async (std::launch::async,
                        [ranking, model_json, layer_idx, in_data, target_data, rows, &candidates, &mutex]
                        {
                            for (int r = 0; r < rows; ++r)
                            {
                                float value {};
                                if (ranking == Ranking::Min_Weights)
                                    value = rank_min_weights (model_json, layer_idx, r);
                                else if (ranking == Ranking::Mean_Activations)
                                    value = rank_mean_activations (model_json, layer_idx, r, in_data);
                                else if (ranking == Ranking::Minimization)
                                    value = rank_minimization (model_json, layer_idx, r, in_data, target_data);

                                std::lock_guard lock { mutex };
                                candidates.push_back (Pruning_Candidate {
                                    .layer = layer_idx,
                                    .row = r,
                                    .value = value,
                                });
                            }
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
    std::cout << "Dense network pruning test\n";

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

        static constexpr auto n_prune = 32;
        model_json = prune (model_json, std::span { pruning_candidates }.subspan (0, n_prune));
        std::cout << "Parameter count: " << count_params (model_json) << '\n';
        Model model { model_json };
        const auto model_out = run_model (model, in_data);
        std::cout << "Prune 1 MSE: " << compute_mse (model_out, target_data) << '\n';
    }

    return 0;
}
