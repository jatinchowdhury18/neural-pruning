#pragma once

#include <juce_core/juce_core.h>
#include <RTNeural/RTNeural.h>
#include <span>

enum class Ranking
{
    Min_Weights = 1,
    Mean_Activations = 2,
    Minimization = 4,
};

struct LSTM_Model
{
    static constexpr int input_size = 1;
    static constexpr int max_hidden_size = 84;
    static constexpr int min_hidden_size = 48;

    template <int hidden_size>
    struct Model
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
        using type = typename concatenator<typename Model_Variant_Builder<hidden_size - 1>::type, Model<hidden_size>>::type;
    };

    template <>
    struct Model_Variant_Builder<min_hidden_size>
    {
        using type = std::variant<Model<min_hidden_size>>;
    };
    using Model_Variant = Model_Variant_Builder<max_hidden_size>::type;

    Model_Variant model_variant {};
    nlohmann::json original_model_json {};
    juce::SpinLock model_loading_mutex {};

    void load (const nlohmann::json& model_json);
    void process (std::span<float> data);
    void prune (int pruned_hidden_size, Ranking ranking);
};
