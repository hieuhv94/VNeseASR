/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "Session.h"

using namespace w2l;
using namespace streaming;
namespace pnk
{
    // Return true if a sentence is complete
    inline bool getChunkTranscription(const std::vector<WordUnit> &wordUnits, std::string &results)
    {

        if (!wordUnits.size())
        {
            return true;
        }
        results.clear();
        for (const auto &wordUnit : wordUnits)
        {
            results += wordUnit.word + " ";
        }
        return false;
    }
    void Session::init(
        std::shared_ptr<const DecoderFactory> decoderFactory,
        fl::lib::text::LexiconDecoderOptions decoderOptions,
        std::shared_ptr<streaming::Sequential> featureModule,
        std::shared_ptr<streaming::Sequential> acousticModule,
        std::shared_ptr<KenLM> kenLM,
        std::shared_ptr<PunctuationAndNer> punc,
        SessionConfig configure,
        plugin_type_t type)
    {
        _dnnModule = std::make_shared<streaming::Sequential>();
        _dnnModule->add(featureModule);
        _dnnModule->add(acousticModule);
        _input = std::make_shared<streaming::ModuleProcessingState>(1);

        _kenLM = kenLM;
        _punc = punc;
        _decoderFactory = decoderFactory;
        _decoderOptions = decoderOptions;
        _minChunkMsec = configure.minChunkMsec;
        _tokensSize = configure.tokensSize;
        _interuptTime = configure.interuptTime;
        _status = READY;
        _type = type;
    }
    void Session::asrProcess(const LoadDataIntoSessionMethod &readDataFunc)
    {
        auto output = _dnnModule->start(_input);
        auto outputBuffer = output->buffer(0);
        bool finish = false;
        int silientTimeMsec = 0;
        bool running = false;
        std::string tmpText = "";
        constexpr const int lookBack = 0;
        constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
        int minChunkSize = _minChunkMsec * kAudioWavSamplingFrequency / 1000;
        auto _decoder = _decoderFactory->createDecoder(_decoderOptions);
        _decoder.start();
        while ((!finish) && (_status == RUNNING))
        {
            std::thread::id this_id = std::this_thread::get_id();
            size_t curChunkSize = readDataFunc(_input->buffer(0));
            if (curChunkSize >= minChunkSize)
            {
                // Hot fix crash bug when websocket diconnect
                if (_status != RUNNING)
                    break;
                _dnnModule->run(_input);
                float *data = outputBuffer->data<float>();
                int size = outputBuffer->size<float>();
                if (data && size > 0)
                {
                    _decoder.run(data, size);
                }
            }
            else
            {
                _dnnModule->finish(_input);
                float *data = outputBuffer->data<float>();
                int size = outputBuffer->size<float>();
                if (data && size > 0)
                {
                    _decoder.run(data, size);
                }
                _decoder.finish();
                finish = true;
            }
            const int chunkSizeMsec = (curChunkSize /
                                       (kAudioWavSamplingFrequency / 1000));
            if (getChunkTranscription(_decoder.getBestHypothesisInWords(lookBack), tmpText))
            {
                silientTimeMsec += chunkSizeMsec;
            }
            else
            {
                if (!running)
                    running = true;
                silientTimeMsec = 0;
                _textCorrectionQueue.push(tmpText);
            }
            // Consume and prune
            const int nFramesOut = outputBuffer->size<float>() / _tokensSize;
            outputBuffer->consume<float>(nFramesOut * _tokensSize);
            _decoder.prune(lookBack);
            if (((silientTimeMsec >= _interuptTime) && running) || (finish && (!tmpText.empty())))
            {
                _decoder.reset();
                _textCorrectionQueue.push("\n");
                tmpText.clear();
                silientTimeMsec = 0;
                running = false;
            }
        }
        // Cannot get input audio so destroy session
        destroy();
    }

    void Session::textCorrectionProcess(const OutputSessionMethod &finalOutputHandle, const OutputSessionMethod &tmpOutputHandle)
    {
        PnkTextCorrector corrector;
        std::string rawText = "";
        while ((_status == RUNNING) || (!_textCorrectionQueue.empty()))
        {
            std::string tmpText = "";
            if(_textCorrectionQueue.pop(tmpText) < 0)
            {
                break;
            }
            else
            {
                if (tmpText.empty())
                    break;
                if (tmpText.find("\n") != std::string::npos)
                {
                    rawText += tmpText;
                    finalOutputHandle(corrector.run(rawText, _kenLM, _punc));
                    rawText.clear();
                }
                else
                {
                    rawText += tmpText;
                    tmpOutputHandle(tmpText);
                }
            }
        }
    }

    void Session::run(const LoadDataIntoSessionMethod &readDataFunc,
                      const OutputSessionMethod &finalOutputHandle,
                      const OutputSessionMethod &tmpOutputHandle)
    {
        _status = RUNNING;
        std::thread asrThread ([readDataFunc, this] () {
            asrProcess(readDataFunc);
        });

        std::thread textCorrectionThread ([finalOutputHandle, tmpOutputHandle, this] () {
            textCorrectionProcess(finalOutputHandle, tmpOutputHandle);
        });
        if (_type == SYNC)
        {
            asrThread.join();
            textCorrectionThread.join();
        } else if (_type == ASYNC)
        {
            asrThread.detach();
            textCorrectionThread.detach();
        }
    }
    void Session::destroy()
    {
        _status = IDLE;
        _textCorrectionQueue.free();
    }
} // namespace pnk