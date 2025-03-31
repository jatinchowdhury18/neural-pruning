#include <chowdsp_buffers/chowdsp_buffers.h>
#include <chowdsp_math/chowdsp_math.h>

#include "neural_pruning_plugin.h"
#include "plugin_editor.h"

Neural_Pruning_Plugin::Neural_Pruning_Plugin()
{
    const auto model_path { std::string { MODELS_DIR } + "/lstm.json" };
    std::ifstream { model_path, std::ifstream::binary } >> lstm_model.original_model_json;
    lstm_model.load (lstm_model.original_model_json);

    for (auto* param : std::initializer_list<juce::RangedAudioParameter*> { state.params.hidden_size.get(), state.params.ranking.get() })
    {
        callbacks += {
            state.addParameterListener (*param,
                                        chowdsp::ParameterListenerThread::MessageThread,
                                        [this]
                                        {
                                            const auto hidden_size = static_cast<int> (state.params.hidden_size->get());
                                            const auto ranking = state.params.ranking->get();
                                            lstm_model.prune (hidden_size, ranking);
                                        }),
        };
    }
}

void Neural_Pruning_Plugin::prepareToPlay (double sample_rate,
                                           int samples_per_block)
{
    const auto os_ratio = sample_rate <= 48000.0 ? 2 : 1;

    const auto mono_spec = juce::dsp::ProcessSpec {
        sample_rate,
        static_cast<uint32_t> (samples_per_block),
        1,
    };
    const auto os_mono_spec = juce::dsp::ProcessSpec {
        sample_rate * os_ratio,
        static_cast<uint32_t> (samples_per_block * os_ratio),
        1,
    };

    upsampler.prepare (mono_spec, os_ratio);
    downsampler.prepare (os_mono_spec, os_ratio);

    dc_blocker.prepare (mono_spec);
    dc_blocker.setCutoffFrequency (10.0f);
}

void Neural_Pruning_Plugin::releaseResources()
{
}

void Neural_Pruning_Plugin::processAudioBlock (juce::AudioBuffer<float>& buffer)
{
    // sum to mono
    chowdsp::BufferView mono_buffer { buffer, 0, -1, 0, 1 };
    chowdsp::BufferMath::sumToMono (buffer, mono_buffer);

    // upsample
    const auto os_buffer = upsampler.process (mono_buffer);

    // process neural network
    lstm_model.process (os_buffer.getWriteSpan (0));

    // downsample
    downsampler.process (os_buffer, mono_buffer);

    // dc blocker
    dc_blocker.processBlock (mono_buffer);

    // separate back out to stereo
    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        chowdsp::BufferMath::copyBufferChannels (mono_buffer, buffer, 0, ch);
}

juce::AudioProcessorEditor* Neural_Pruning_Plugin::createEditor()
{
    return new Plugin_Editor { *this };
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Neural_Pruning_Plugin {};
}
