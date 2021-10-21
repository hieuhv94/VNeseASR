/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "plugins/files/FilePlugin.h"
namespace pnk {
w2l::streaming::LoadDataIntoSessionMethod FilePlugin::getLoadDataFunc(
    std::ifstream& inputAudioStream,
    int minChunkMsec
)
{
    return [&inputAudioStream, minChunkMsec, this] (std::shared_ptr<w2l::streaming::IOBuffer> input)
    {
        constexpr const float kMaxUint16 = static_cast<float>(0x8000);
            constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
            const int minChunkSize = minChunkMsec * kAudioWavSamplingFrequency / 1000;
                return readTransformStreamIntoBuffer<int16_t, float>(
                inputAudioStream, input, minChunkSize, [](int16_t i) -> float {
                    return static_cast<float>(i) / kMaxUint16;
                });
    };
}

w2l::streaming::OutputSessionMethod FilePlugin::getOutputFunc(std::string& output) {
    return [&output] (std::string text)
    {
        output += text + "\n";
    };
}

void FilePlugin::runSync(const char* audioPath, std::string& output)
{
    session_id_t sid = _service->createSession(SYNC);
    std::ifstream inputAudioStream(audioPath, std::ios::binary);
    _service->sessionRun(sid, getLoadDataFunc(inputAudioStream, _minChunkMsec), getOutputFunc(output), ([](std::string text){
        // Don't anything
    }));
    _service->sessionEnd(sid);
}

void FilePlugin::runInteractiveMode()
{
    while(true)
    {
    std::string audioPath, output;
    // Read input file fron keyboards
    std::cout << "Enter audio file name: ";
    std::cin >> audioPath;
    session_id_t sid = _service->createSession(SYNC);
    std::ifstream inputAudioStream(audioPath, std::ios::binary);
    _service->sessionRun(sid, getLoadDataFunc(inputAudioStream, _minChunkMsec), getOutputFunc(output), ([](std::string text){
        // Don't anything
    }));
    std::cout << "Transcripts: " << output << std::endl;
    _service->sessionEnd(sid);
    }
}
}