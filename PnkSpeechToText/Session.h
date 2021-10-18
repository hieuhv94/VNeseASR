/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/lib/text/decoder/Decoder.h"
#include "models/asr/nn/nn.h"
#include "utils/common.h"
#include "config.h"
#include <thread>
using namespace w2l;
using namespace streaming;
namespace pnk
{
    typedef uint32_t session_id_t;
    typedef enum session_status_e
    {
        IDLE = 0,
        READY,
        RUNNING,
        STOP
    } session_status_t;

    typedef enum plugin_type_t
    {
        SYNC,
        ASYNC
    } plugin_type_t;
    class Session
    {
    public:
        Session(session_id_t sId) : _sessionId(sId), _status(IDLE){};
        ~Session(){};
        void init(
            std::shared_ptr<const DecoderFactory> decoderFactory,
            fl::lib::text::LexiconDecoderOptions decoderOptions,
            std::shared_ptr<streaming::Sequential> featureModule,
            std::shared_ptr<streaming::Sequential> acousticModule,
            std::shared_ptr<KenLM> kenLM,
            std::shared_ptr<PunctuationAndNer> punc,
            SessionConfig configure,
            plugin_type_t type);
        void destroy();
        void run(const LoadDataIntoSessionMethod& readDataFunc,
            const OutputSessionMethod &finalOutputHandle,
            const OutputSessionMethod &tmpOutputHandle);

    private:
        const session_id_t _sessionId;
        plugin_type_t _type;
        session_status_t _status;
        std::shared_ptr<streaming::Sequential> _dnnModule;
        std::shared_ptr<streaming::ModuleProcessingState> _input;
        // w2l::streaming::Decoder _decoder;
        std::shared_ptr<const DecoderFactory> _decoderFactory;
        fl::lib::text::LexiconDecoderOptions _decoderOptions;
        int _minChunkMsec;
        int _interuptTime;
        std::shared_ptr<KenLM> _kenLM;
        std::shared_ptr<PunctuationAndNer> _punc;
        Queue<std::string> _textCorrectionQueue;
        size_t _tokensSize;
        // LoadDataIntoSessionMethod _loadData;
        // OutputSessionMethod _outputFunc;
        void asrProcess(const LoadDataIntoSessionMethod& readDataFunc);
        void textCorrectionProcess(const OutputSessionMethod &finalOutputHandle, const OutputSessionMethod &tmpOutputHandle);
    };
} // namespace pnk