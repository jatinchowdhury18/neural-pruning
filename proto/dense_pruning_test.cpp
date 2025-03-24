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
    // static constexpr int num_samples = 5'000;

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
    const auto model_path { std::string { TRAIN_DIR } + "/dense.json" };
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
    static constexpr auto num_layers = 8;
    static constexpr auto layer_width = 64;

    std::vector<RTNeural::Dense<float>> dense_layers {};
    RTNeural::ReLuActivation<float> relu_activation { layer_width };
    alignas (16) std::array<std::array<float, layer_width>, num_layers + 1> layer_io {};

    explicit Model (const nlohmann::json& model_json)
    {
        dense_layers.reserve (num_layers + 1);

        assert (model_json["in_shape"].back() == 1);
        int in_size = 1;
        for (auto& layer : model_json["layers"])
        {
            if (layer["type"] == "activation")
                continue;

            // std::cout << "Loading layer " << layer.dump() << '\n';
            const auto out_size = layer["shape"].back().get<int>();
            // std::cout << "Dense {" << in_size << "," << out_size << "}\n";
            auto& dense_layer = dense_layers.emplace_back (in_size, out_size);
            RTNeural::json_parser::loadDense<float> (dense_layer, layer["weights"]);
            in_size = out_size;
        }
        assert (in_size == 1);
    }

    float forward (const float* in) noexcept
    {
        dense_layers.front().forward (in, layer_io.front().data());

        for (int i = 0; i < num_layers; ++i)
        {
            relu_activation.forward (layer_io[i].data(), layer_io[i].data());
            dense_layers[i + 1].forward (layer_io[i].data(), layer_io[i + 1].data());
        }

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

template <typename Model_Type>
static std::vector<float> run_model (Model_Type& model, std::span<const float> input, bool verbose = true, int num_iters = 1)
{
    std::vector<float> out (input.size());

    const auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i)
    {
        for (size_t n = 0; n < input.size(); ++n)
            out[n] = model.forward (&input[n]);
    }

    const auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto test_duration_seconds = std::chrono::duration<float> { duration }.count();
    if (verbose)
    {
        std::cout << "Inference Time: " << test_duration_seconds << " seconds" << std::endl;
        const auto audio_time = (float) (input.size() * num_iters) / 96'000.0f;
        const auto real_time_factor = audio_time / test_duration_seconds;
        std::cout << "Real-Time Factor: " << real_time_factor << std::endl;
    }

    return out;
}

struct Pruning_Candidate
{
    int layer {};
    int row { -1 };
    int column { -1 };
    float value { 0.0f };
};

static nlohmann::json prune (nlohmann::json model_json,
                             std::span<Pruning_Candidate> candidates_to_prune,
                             int start,
                             int num)
{
    int count = 0;
    for (int prune_idx = start; prune_idx < start + num; ++prune_idx)
    {
        auto& to_prune = candidates_to_prune[prune_idx];
        const auto erase_row = [&] (int layer_idx, int row_idx)
        {
            for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
            {
                auto& to_fix = candidates_to_prune[fix_idx];
                if (to_fix.layer == layer_idx)
                {
                    if (to_fix.row == row_idx)
                        to_fix.row = -1;
                    if (to_fix.row > row_idx)
                        to_fix.row--;
                }
            }
        };

        const auto erase_col = [&] (int layer_idx, int col_idx)
        {
            for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
            {
                auto& to_fix = candidates_to_prune[fix_idx];
                if (to_fix.layer == layer_idx)
                {
                    if (to_fix.column == col_idx)
                        to_fix.column = -1;

                    if (to_fix.column > col_idx)
                        to_fix.column--;
                }
            }
        };

        auto& layers = model_json["layers"];
        for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
        {
            if (layers.at (layer_idx)["type"] == "activation")
                continue;

            // Convention:
            // Rows -> out size
            // Cols -> in size

            if (to_prune.layer == layer_idx)
            {
                if (to_prune.column >= 0) // pruning a column
                {
                    to_prune.layer -= 2;
                    to_prune.row = to_prune.column;
                    if (to_prune.layer < 0)
                        break;
                }

                if (to_prune.row >= 0) // pruning a row
                {
                    auto& layer = layers.at (to_prune.layer);
                    auto& weights = layer["weights"].at (0);
                    auto& biases = layer["weights"].at (1);

                    if (weights[0].size() <= 1)
                        break;

                    for (auto& w : weights)
                        w.erase (to_prune.row);
                    biases.erase (to_prune.row);

                    auto next_layer_idx = to_prune.layer + 2;
                    if (next_layer_idx < layers.size())
                    {
                        auto& next_layer = layers.at (next_layer_idx);
                        auto& next_weights = next_layer["weights"].at (0);
                        next_weights.erase (to_prune.row);
                        erase_col (next_layer_idx, to_prune.row);
                    }

                    erase_row (to_prune.layer, to_prune.row);
                    layer["shape"].back() = layer["shape"].back().get<int>() - 1;

                    count++;
                }

                break;
            }
        }
    }

    std::cout << "Pruning " << count << " structural elements...\n";

    return model_json;
}

static float rank_min_weights (const nlohmann::json& model_json,
                               int layer_idx,
                               int row,
                               int col)
{
    auto& layers = model_json["layers"].at (layer_idx);
    auto& weights = layers["weights"].at (0);

    float square_sum = 0.0f;
    if (row >= 0)
    {
        for (auto& w : weights)
        {
            auto v = w.at (row).get<float>();
            square_sum += v * v;
        }
    }
    else if (col >= 0)
    {
        for (const auto& w : weights.at (col))
        {
            auto v = w.get<float>();
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
        [[maybe_unused]] auto _ = model.forward (&in_data[n]);
        activation_out[n] = model.layer_io[layer_idx / 2][row];
    }

    const auto mean = compute_mean (activation_out);
    const auto variance = compute_stddev (activation_out, mean);

    return variance;
}

static float rank_minimization (nlohmann::json model_json,
                                int layer_idx,
                                int row,
                                int col,
                                std::span<const float> in_data,
                                std::span<const float> target_data)
{
    auto& layers = model_json["layers"].at (layer_idx);
    auto& weights = layers["weights"].at (0);
    if (row >= 0)
    {
        for (auto& w : weights)
            w.at (row) = 0.0f;
    }
    else if (col >= 0)
    {
        for (auto& w : weights.at (col))
            w = 0.0f;
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
    int cols = 1;

    auto& layers = model_json["layers"];
    for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
    {
        auto& layer = layers.at (layer_idx);
        if (layer["type"] == "activation")
            continue;

        const auto rows = layer["shape"].back().get<int>();

        futures.push_back (std::async (std::launch::async,
                                       [ranking, model_json, layer_idx, in_data, target_data, rows, cols, &candidates, &mutex]
                                       {
                                           if (rows > 1)
                                           {
                                               for (int r = 0; r < rows; ++r)
                                               {
                                                   float value {};
                                                   if (ranking == Ranking::Min_Weights)
                                                       value = rank_min_weights (model_json, layer_idx, r, -1);
                                                   else if (ranking == Ranking::Mean_Activations)
                                                       value = rank_mean_activations (model_json, layer_idx, r, in_data);
                                                   else if (ranking == Ranking::Minimization)
                                                       value = rank_minimization (model_json, layer_idx, r, -1, in_data, target_data);

                                                   std::lock_guard lock { mutex };
                                                   candidates.push_back (Pruning_Candidate {
                                                       .layer = layer_idx,
                                                       .row = r,
                                                       .column = -1,
                                                       .value = value,
                                                   });
                                               }
                                           }
                                           if (cols > 1)
                                           {
                                               for (int c = 0; c < cols; ++c)
                                               {
                                                   float value {};
                                                   if (ranking == Ranking::Min_Weights)
                                                       value = rank_min_weights (model_json, layer_idx, -1, c);
                                                   else if (ranking == Ranking::Minimization)
                                                       value = rank_minimization (model_json, layer_idx, -1, c, in_data, target_data);
                                                   else
                                                       break;

                                                   std::lock_guard lock { mutex };
                                                   candidates.push_back (Pruning_Candidate {
                                                       .layer = layer_idx,
                                                       .row = -1,
                                                       .column = c,
                                                       .value = value,
                                                   });
                                               }
                                           }
                                       }));

        cols = rows; // for next layer
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

    // {
    //     std::cout << "Parameter count: " << count_params (model_json) << '\n';
    //     Model model { model_json };
    //     const auto model_out = run_model (model, in_data);
    //     std::cout << "Post-Training MSE: " << compute_mse (model_out, target_data) << '\n';
    // }

    // const auto ranking = Ranking::Min_Weights;
    const auto ranking = Ranking::Mean_Activations;
    // const auto ranking = Ranking::Minimization;
    auto pruning_candidates = rank_pruning_candidates (model_json, ranking, in_data, target_data);
    std::cout << "# Pruning Candidates: " << pruning_candidates.size() << '\n';

    int iter = 0;
    do
    {
        std::cout << "Parameter count: " << count_params (model_json) << '\n';
        Model model { model_json };
        const auto model_out = run_model (model, in_data, true, 10);
        std::cout << "Prune " << iter << " MSE: " << compute_mse (model_out, target_data) << '\n';

        static constexpr auto n_prune = 28;
        model_json = prune (model_json, pruning_candidates, n_prune * iter, n_prune);

        using namespace std::chrono_literals;
        std::this_thread::sleep_for (1000ms);
    } while (++iter <= 8);

    return 0;
}
