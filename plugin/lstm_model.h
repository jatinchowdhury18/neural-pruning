#pragma once

#include <juce_core/juce_core.h>
#include <RTNeural/RTNeural.h>
#include <span>

struct LSTM_Model
{
    static constexpr int input_size = 2;
    static constexpr int max_hidden_size = 24;
    static constexpr int min_hidden_size = 2;

    template <int hidden_size>
    using Model = RTNeural::ModelT<float,
                                   input_size,
                                   1,
                                   RTNeural::LSTMLayerT<float, input_size, hidden_size>,
                                   RTNeural::DenseT<float, hidden_size, 1>>;

    template <typename T, typename... Args> struct concatenator;
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

    template<>
    struct Model_Variant_Builder<2>
    {
        using type = std::variant<Model<2>>;
    };
    using Model_Variant = Model_Variant_Builder<max_hidden_size>::type;

    juce::SpinLock model_loading_mutex {};
    Model_Variant model_variant {};
    nlohmann::json model_json_original;

    void load_model (const nlohmann::json& model_json);
    void reload_original_model();
    void load_model (const nlohmann::json& state_dict, int hidden_size);
    void prune_model (float pruning_error_threshold);
    void process (std::span<float> data, float param);

    int current_hidden_size {};
    nlohmann::json model_state_dict {};
    void find_pruning_candidate();
    void prune_channel (int channel_idx);
    chowdsp::Broadcaster<void (int channel, float rms_error)> new_pruning_candidate {};
    chowdsp::Broadcaster<void()> model_changed {};
};
