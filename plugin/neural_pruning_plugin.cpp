#include <chowdsp_gui/chowdsp_gui.h>
#include <chowdsp_buffers/chowdsp_buffers.h>
#include <chowdsp_math/chowdsp_math.h>

#include "neural_pruning_plugin.h"

Neural_Pruning_Plugin::Neural_Pruning_Plugin()
{
    const auto model_path { std::string { MODELS_DIR } + "/fuzz_2.json" };
    nlohmann::json model_json {};
    std::ifstream { model_path, std::ifstream::binary } >> model_json;
    lstm_model.load_model (model_json);
}

void Neural_Pruning_Plugin::prepareToPlay (double sample_rate,
                                           int samples_per_block)
{
    juce::ignoreUnused (sample_rate, samples_per_block);
}

void Neural_Pruning_Plugin::releaseResources()
{
}

void Neural_Pruning_Plugin::processAudioBlock (juce::AudioBuffer<float>& buffer)
{
    chowdsp::BufferView mono_buffer { buffer, 0, -1, 0, 1 };
    chowdsp::BufferMath::sumToMono (buffer, mono_buffer);

    const auto param = state.params.gain->getCurrentValue();
    lstm_model.process (mono_buffer.getWriteSpan (0), param);

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        chowdsp::BufferMath::copyBufferChannels (mono_buffer, buffer, 0, ch);
}

juce::AudioProcessorEditor* Neural_Pruning_Plugin::createEditor()
{
    return new chowdsp::ParametersViewEditor { *this };
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Neural_Pruning_Plugin {};
}
