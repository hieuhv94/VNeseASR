/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef MICROPHONE_PLUGIN_H
#define MICROPHONE_PLUGIN_H
#include "PnkS2t.h"
#include "plugins/AudioBuffer.h"
#include "plugins/Resample.h"
namespace pnk
{
    class PnkSpeechToText;
    class MicroPlugin
    {
    public:
        MicroPlugin(PnkSpeechToText *service, int minChunkMsec) : _service(service), _minChunkMsec(minChunkMsec), _buffer(AUDIO_BUFFER_MAX_SIZE){};
        ~MicroPlugin() {};
        void run(const char *device, unsigned int sampleRate = 44100, int channels = 2);
        
    private:
        PnkSpeechToText *_service;
        int _minChunkMsec;
        AudioBuffer _buffer;
        w2l::streaming::LoadDataIntoSessionMethod getLoadDataFunc(int minChunkMsec);
		// w2l::streaming::OutputSessionMethod getFinalOutputFunc(session_id_t sid, WebsocketServer* server);
		w2l::streaming::OutputSessionMethod getFinalOutputFunc();
    };
} // namespace pnk
#endif // IODEVICE_PLUGIN_H