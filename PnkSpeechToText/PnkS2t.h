/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef MANAGER_H
#define MANAGER_H
#include "Modules.h"
#include "config.h"
#include "Session.h"
#include <unordered_map>
#include "plugins/files/FilePlugin.h"
#include "plugins/websocket/WebSocketPlugin.h"
#include "plugins/microphone/MicroPlugin.h"
#include "plugins/socket/SocketPlugin.h"
namespace pnk
{
    class FilePlugin;
    class WebSocketPlugin;
    class MicroPlugin;
    class SocketPlugin;
    class PnkSpeechToText {
        public:
        PnkSpeechToText(std::string configPath);
        session_id_t createSession(plugin_type_t type);
        void sessionRun(const session_id_t sid,  const LoadDataIntoSessionMethod &readDataFunc,
                        const OutputSessionMethod &finalOutputHandle,
                        const OutputSessionMethod &tmpOutputHandle);
        void sessionEnd(const session_id_t sid);
        FilePlugin* getFilePlugin();
        WebSocketPlugin* getWebSocketPlugin();
        MicroPlugin* getDevicePlugin();
        SocketPlugin* getSocketPlugin();
        private:
        ServiceConfig _configure;
        AsrModules _asrModules;
        std::mutex _sessionMutex;
        session_id_t _maxId = 0;
        TextCorrectionModules _textCorrectionModules;
        std::unordered_map<session_id_t, Session*> _sessionList;
        session_id_t newId();
    };
}
#endif //MANAGER_H