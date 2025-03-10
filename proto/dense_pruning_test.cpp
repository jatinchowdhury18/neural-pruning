#include <iostream>
#include <random>
#include <sndfile.h>

#include <RTNeural/RTNeural.h>

static std::tuple<std::vector<float>, std::vector<float>> get_audio_data()
{
    // re-use the same data that we used for training
    static constexpr int seek_offset = 2'000'000;
    static constexpr int num_samples = 1'471'622;

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
        // this file is stereo, but we're only going to use te left channel
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
    std::array<std::array<float, layer_width>, num_layers + 1> layer_io {};

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

template <typename Model_Type>
static std::vector<float> run_model (Model_Type& model, std::span<const float> input)
{
    std::vector<float> out (input.size());

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t n = 0; n < input.size(); ++n)
        out[n] = model.forward (&input[n]);

    const auto duration = std::chrono::high_resolution_clock::now() - start;
    const auto test_duration_seconds = std::chrono::duration<float> {  duration }.count();
    std::cout << "Inference Time: " << test_duration_seconds << " seconds" << std::endl;


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
                             std::span<Pruning_Candidate> candidates_to_prune)
{
    std::cout << "Pruning " << candidates_to_prune.size() << " structural elements...\n";

    for (int prune_idx = 0; prune_idx < candidates_to_prune.size(); ++prune_idx)
    {
        const auto& to_prune = candidates_to_prune[prune_idx];
        const auto erase_row = [&] (int layer_idx, int row_idx)
        {
            for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
            {
                auto& to_fix = candidates_to_prune[fix_idx];
                if (to_fix.layer == layer_idx && to_fix.row > row_idx)
                    to_fix.row--;
            }
        };

        const auto erase_col = [&] (int layer_idx, int col_idx)
        {
            for (int fix_idx = prune_idx; fix_idx < candidates_to_prune.size(); ++fix_idx)
            {
                auto& to_fix = candidates_to_prune[fix_idx];
                if (to_fix.layer == layer_idx && to_fix.column > col_idx)
                    to_fix.column--;
            }
        };

        auto& layers = model_json["layers"];
        for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
        {
            auto& layer = layers.at (layer_idx);
            if (layer["type"] == "activation")
                continue;

            // Convention:
            // Rows -> out size
            // Cols -> in size

            if (to_prune.layer == layer_idx)
            {
                if (to_prune.row >= 0) // pruning a row
                {
                    auto& weights = layer["weights"].at (0);
                    auto& biases = layer["weights"].at (1);
                    for (auto& w : weights)
                        w.erase (to_prune.row);
                    erase_row (layer_idx, to_prune.row);
                    biases.erase (to_prune.row);
                    layer["shape"].back() = layer["shape"].back().get<int>() - 1;

                    auto next_layer_idx = layer_idx + 2;
                    if (next_layer_idx < layers.size())
                    {
                        auto& next_layer = layers.at (next_layer_idx);
                        auto& next_weights = next_layer["weights"].at (0);
                        next_weights.erase (to_prune.row);
                        erase_col (next_layer_idx, to_prune.row);
                    }
                }
                else if (to_prune.column >= 0) // pruning a column
                {
                    auto prev_layer_idx = layer_idx - 2;
                    if (prev_layer_idx >= 0)
                    {
                        auto& prev_layer = layers.at (prev_layer_idx);
                        auto& weights = prev_layer["weights"].at (0);
                        auto& biases = prev_layer["weights"].at (1);
                        for (auto& w : weights)
                            w.erase (to_prune.column);
                        erase_row (prev_layer_idx, to_prune.column);
                        biases.erase (to_prune.column);
                        prev_layer["shape"].back() = prev_layer["shape"].back().get<int>() - 1;
                    }

                    auto& weights = layer["weights"].at (0);
                    weights.erase (to_prune.column);
                    erase_col (layer_idx, to_prune.column);
                }

                break;
            }
        }
    }

    return model_json;
}

static float rank (nlohmann::json model_json,
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

static auto rank_pruning_candidates (nlohmann::json model_json)
{
    std::vector<Pruning_Candidate> candidates {};
    int cols = 1;

    auto& layers = model_json["layers"];
    for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx)
    {
        auto& layer = layers.at (layer_idx);
        if (layer["type"] == "activation")
            continue;

        const auto rows = layer["shape"].back().get<int>();

        if (rows > 1)
        {
            for (int r = 0; r < rows; ++r)
            {
                candidates.push_back (Pruning_Candidate {
                    .layer = layer_idx,
                    .row = r,
                    .column = -1,
                    .value = rank (model_json, layer_idx, r, -1),
                });
            }
        }
        if (cols > 1)
        {
            for (int c = 0; c < cols; ++c)
            {
                candidates.push_back (Pruning_Candidate {
                    .layer = layer_idx,
                    .row = -1,
                    .column = c,
                    .value = rank (model_json, layer_idx, -1, c),
                });
            }
        }

        cols = rows; // for next layer
    }

    std::sort (candidates.begin(),
               candidates.end(),
               [] (const Pruning_Candidate& a, const Pruning_Candidate& b)
               {
                   return a.value < b.value;
               });

    return candidates;
}

int main()
{
    std::cout << "Dense network pruning test\n";

    const auto [in_data, target_data] = get_audio_data();
    auto model_json = get_model_json();

    {
        Model model { model_json };
        const auto model_out = run_model (model, in_data);
        std::cout << "Post-Training MSE: " << compute_mse (model_out, target_data) << '\n';
    }

    {
        auto pruning_candidates = rank_pruning_candidates (model_json);
        std::cout << "# Pruning Candidates: " << pruning_candidates.size() << '\n';

        static constexpr auto n_prune = 96;
        model_json = prune (model_json, std::span { pruning_candidates }.subspan (0, n_prune));
        Model model { model_json };
        const auto model_out = run_model (model, in_data);
        std::cout << "Prune 1 MSE: " << compute_mse (model_out, target_data) << '\n';
    }

    return 0;
}
