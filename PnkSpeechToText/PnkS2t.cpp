/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PnkS2t.h"
namespace pnk
{
    PnkSpeechToText::PnkSpeechToText(const std::string configPath)
    {
        parserConfigFile(configPath, _configure);
        // Create ASR modules
        _asrModules.initial(_configure.asr);
        _textCorrectionModules.initial(_configure.textCorrection);
        _configure.session.tokensSize = _asrModules.tokensSize();
    }
    session_id_t PnkSpeechToText::newId()
    {
        while(_sessionList.find(_maxId) != _sessionList.end())
        {
            _maxId++;
        }
        return _maxId++;
    }
    session_id_t PnkSpeechToText::createSession(plugin_type_t type)
    {
        _sessionMutex.lock();   
        session_id_t sid = newId();
        if (sid < 0)
        {
            // Get session id failed
        }
        Session* newSession = new Session(sid);
        newSession->init(_asrModules.decoderFactory,
                        _asrModules.decoderOptions,
                        _asrModules.featureModule,
                        _asrModules.acousticModule,
                        _textCorrectionModules.kenLM,
                        _textCorrectionModules.predPunc,
                        _configure.session,
                        type);
        
        _sessionList[sid] = newSession;
        _sessionMutex.unlock();
        return sid;
    }

    void PnkSpeechToText::sessionRun (const session_id_t sid,
                        const LoadDataIntoSessionMethod &readDataFunc,
                        const OutputSessionMethod &finalOutputHandle,
                        const OutputSessionMethod &tmpOutputHandle) {
        _sessionList[sid]->run(readDataFunc, finalOutputHandle, tmpOutputHandle);
    }

    void PnkSpeechToText::sessionEnd (const session_id_t sid) {
        Session* session = _sessionList[sid];
        session->destroy();
        _sessionList.erase(sid);
        delete session;
    }

    FilePlugin* PnkSpeechToText::getFilePlugin()
    {
        FilePlugin* filePlugin = new FilePlugin(this, _configure.session.minChunkMsec);
        return filePlugin;
    }
    WebSocketPlugin* PnkSpeechToText::getWebSocketPlugin()
    {
        WebSocketPlugin* wsPlugin = new WebSocketPlugin(this, _configure.session.minChunkMsec);
        return wsPlugin;
    }
    MicroPlugin* PnkSpeechToText::getDevicePlugin()
    {
        MicroPlugin* devPlugin = new MicroPlugin(this, _configure.session.minChunkMsec);
        return devPlugin;
    }
    SocketPlugin* PnkSpeechToText::getSocketPlugin()
    {
        SocketPlugin* socketPlugin = new SocketPlugin(this, _configure.session.minChunkMsec);
        return socketPlugin;
    }
}