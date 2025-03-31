#include "lstm_model.h"

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

void LSTM_Model::load (const nlohmann::json& model_json)
{
    juce::SpinLock::ScopedLockType model_loading_lock { model_loading_mutex };

    const auto current_hidden_size = model_json["layers"][0]["shape"].back().get<int>();

    for_each_index (
        [this, current_hidden_size] (auto i)
        {
            if (i == current_hidden_size)
            {
                chowdsp::log ("Loading model with hidden size: {}", current_hidden_size);
                model_variant.emplace<Model<i>>();
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

void LSTM_Model::process (std::span<float> data)
{
    juce::SpinLock::ScopedTryLockType model_loading_lock { model_loading_mutex };
    if (! model_loading_lock.isLocked())
        return;

    std::visit (
        [data] (auto& model)
        {
            for (size_t n = 0; n < data.size(); ++n)
            {
                Eigen::Matrix<float, 1, 1> in { data[n] };
                model.lstm.forward (in);
                model.dense.forward (model.lstm.outs);
                data[n] = model.dense.outs (0);
            }
        },
        model_variant);
}

struct Pruning_Candidate
{
    int idx {};
    float value { 0.0f };
};

// clang-format off
std::array<Pruning_Candidate, 84> min_weights_pruning_candidates {
    Pruning_Candidate { 72, 2.13087 }, { 43, 2.16598 }, { 50, 2.21253 }, { 26, 2.24956 }, { 30, 2.28832 }, { 29, 2.29673 }, { 0, 2.34003 }, { 22, 2.341 }, { 46, 2.35479 }, { 35, 2.356 }, { 41, 2.37336 }, { 3, 2.37712 }, { 56, 2.38788 }, { 24, 2.39144 }, { 5, 2.40586 }, { 21, 2.41391 }, { 59, 2.41807 }, { 57, 2.42462 }, { 53, 2.4294 }, { 31, 2.42972 }, { 1, 2.44275 }, { 52, 2.45693 }, { 70, 2.46812 }, { 82, 2.47018 }, { 55, 2.48423 }, { 73, 2.48888 }, { 14, 2.49051 }, { 40, 2.53333 }, { 16, 2.53497 }, { 28, 2.54347 }, { 69, 2.55042 }, { 74, 2.55634 }, { 44, 2.59142 }, { 78, 2.60429 }, { 54, 2.65403 }, { 6, 2.65591 }, { 18, 2.67383 }, { 61, 2.67644 }, { 38, 2.69066 }, { 23, 2.71 }, { 7, 2.72789 }, { 81, 2.75511 }, { 8, 2.762 }, { 67, 2.76214 }, { 33, 2.77666 }, { 60, 2.80929 }, { 37, 2.81476 }, { 77, 2.8367 }, { 17, 2.85831 }, { 34, 2.91079 }, { 76, 2.93731 }, { 25, 3.01386 }, { 32, 3.01935 }, { 20, 3.03626 }, { 10, 3.03828 }, { 13, 3.06551 }, { 11, 3.08843 }, { 75, 3.11159 }, { 36, 3.19877 }, { 42, 3.46591 }, { 51, 3.47466 }, { 79, 3.48881 }, { 71, 3.49033 }, { 83, 3.5835 }, { 68, 3.59203 }, { 62, 3.59366 }, { 27, 3.66916 }, { 64, 3.70499 }, { 45, 3.73703 }, { 63, 3.74909 }, { 48, 4.02339 }, { 2, 4.03817 }, { 58, 5.0306 }, { 9, 5.33691 }, { 12, 5.35558 }, { 39, 5.43005 }, { 49, 5.57972 }, { 80, 5.9916 }, { 15, 6.3741 }, { 66, 6.5782 }, { 19, 6.59422 }, { 4, 7.28951 }, { 47, 10.3351 }, { 65, 14.7341 }
};

std::array<Pruning_Candidate, 84> mean_activations_pruning_candidates {
        Pruning_Candidate { 29, 0.0132998 }, { 64, 0.0133761 }, { 27, 0.0135199 }, { 79, 0.0165395 }, { 8, 0.0179986 }, { 9, 0.0191356 }, { 82, 0.0210732 }, { 43, 0.0235517 }, { 80, 0.0359793 }, { 6, 0.0373224 }, { 25, 0.0378705 }, { 50, 0.0381939 }, { 73, 0.0400595 }, { 83, 0.0403435 }, { 58, 0.0424952 }, { 55, 0.0431771 }, { 20, 0.044029 }, { 45, 0.044879 }, { 3, 0.0451507 }, { 17, 0.0471593 }, { 63, 0.0485305 }, { 61, 0.0490895 }, { 69, 0.0518686 }, { 26, 0.0538845 }, { 11, 0.0543172 }, { 74, 0.0547031 }, { 31, 0.0562666 }, { 33, 0.0562711 }, { 0, 0.0571548 }, { 14, 0.0586037 }, { 68, 0.0586206 }, { 71, 0.0587797 }, { 1, 0.0594811 }, { 24, 0.0596107 }, { 38, 0.0618862 }, { 22, 0.0649005 }, { 72, 0.0663892 }, { 41, 0.0688667 }, { 53, 0.069353 }, { 77, 0.0696635 }, { 15, 0.073035 }, { 57, 0.0735088 }, { 23, 0.0746521 }, { 16, 0.0750336 }, { 37, 0.0808879 }, { 32, 0.0828361 }, { 48, 0.0832864 }, { 35, 0.0839906 }, { 42, 0.0842802 }, { 51, 0.0850634 }, { 62, 0.0858261 }, { 40, 0.0859704 }, { 21, 0.0868093 }, { 70, 0.0872767 }, { 34, 0.0903506 }, { 44, 0.0920573 }, { 18, 0.0922123 }, { 59, 0.0924761 }, { 10, 0.0930473 }, { 7, 0.0960455 }, { 46, 0.0964477 }, { 81, 0.0971485 }, { 2, 0.0972837 }, { 60, 0.0988093 }, { 78, 0.103287 }, { 30, 0.105419 }, { 4, 0.106479 }, { 28, 0.108821 }, { 36, 0.108999 }, { 76, 0.109056 }, { 75, 0.119486 }, { 5, 0.122386 }, { 54, 0.122934 }, { 13, 0.123083 }, { 56, 0.125901 }, { 52, 0.128307 }, { 67, 0.130226 }, { 65, 0.161434 }, { 19, 0.162559 }, { 39, 0.170058 }, { 47, 0.191377 }, { 66, 0.197024 }, { 12, 0.210609 }, { 49, 0.243554 }
};

std::array<Pruning_Candidate, 84> minimization_pruning_candidates {
    Pruning_Candidate { 19, 0.00606051 }, { 4, 0.00620896 }, { 66, 0.0062759 }, { 70, 0.00639654 }, { 71, 0.00643949 }, { 58, 0.00650497 }, { 65, 0.00652915 }, { 63, 0.00657849 }, { 51, 0.00686096 }, { 36, 0.00688572 }, { 73, 0.00691499 }, { 27, 0.00697249 }, { 76, 0.00698026 }, { 33, 0.00698058 }, { 67, 0.00701391 }, { 79, 0.00701659 }, { 43, 0.00705165 }, { 0, 0.00705538 }, { 46, 0.00705574 }, { 17, 0.00706225 }, { 48, 0.00707649 }, { 82, 0.00708469 }, { 25, 0.00709425 }, { 64, 0.00709634 }, { 24, 0.00710677 }, { 45, 0.00710999 }, { 75, 0.00711259 }, { 38, 0.00714537 }, { 54, 0.00716586 }, { 23, 0.00719438 }, { 3, 0.00719887 }, { 78, 0.00720026 }, { 29, 0.00720928 }, { 50, 0.00721218 }, { 7, 0.00721311 }, { 8, 0.00722475 }, { 80, 0.00723986 }, { 22, 0.00724129 }, { 30, 0.0072538 }, { 5, 0.007261 }, { 11, 0.00727795 }, { 9, 0.00728314 }, { 74, 0.00732124 }, { 10, 0.00733704 }, { 83, 0.00737063 }, { 55, 0.00737387 }, { 69, 0.00738152 }, { 72, 0.00738244 }, { 20, 0.00740705 }, { 41, 0.00746363 }, { 61, 0.00756913 }, { 57, 0.00772047 }, { 6, 0.00772506 }, { 77, 0.00777376 }, { 15, 0.0079304 }, { 16, 0.00801656 }, { 62, 0.00804156 }, { 68, 0.00804431 }, { 18, 0.0080775 }, { 26, 0.00812511 }, { 81, 0.00815514 }, { 2, 0.00841405 }, { 1, 0.00842687 }, { 14, 0.00852634 }, { 31, 0.00859142 }, { 60, 0.00860939 }, { 35, 0.00864655 }, { 13, 0.00867723 }, { 34, 0.00872107 }, { 47, 0.00874136 }, { 37, 0.00882984 }, { 32, 0.0088991 }, { 49, 0.00906185 }, { 59, 0.00915053 }, { 40, 0.00918585 }, { 56, 0.00924993 }, { 21, 0.00936529 }, { 53, 0.00968861 }, { 44, 0.00991494 }, { 42, 0.0100489 }, { 52, 0.0101871 }, { 12, 0.0113176 }, { 39, 0.0123618 }, { 28, 0.0158604 }
};
// clang-format on

static nlohmann::json prune (nlohmann::json model_json,
                             std::span<Pruning_Candidate> candidates_to_prune,
                             int start,
                             int num)
{
    for (int prune_idx = start; prune_idx < start + num; ++prune_idx)
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
    }

    return model_json;
}

void LSTM_Model::prune (int pruned_hidden_size, Ranking ranking)
{
    chowdsp::log ("Pruning to hidden size {} with ranking {}",
                  pruned_hidden_size,
                  magic_enum::enum_name (ranking));

    std::span<Pruning_Candidate> pruning_candidates {};
    if (ranking == Ranking::Min_Weights)
        pruning_candidates = min_weights_pruning_candidates;
    else if (ranking == Ranking::Mean_Activations)
        pruning_candidates = mean_activations_pruning_candidates;
    else if (ranking == Ranking::Minimization)
        pruning_candidates = minimization_pruning_candidates;

    const auto pruned_model = ::prune (original_model_json, pruning_candidates, 0, max_hidden_size - pruned_hidden_size);
    load (pruned_model);
}
