/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef WEBSOCKET_PLUGIN_H
#define WEBSOCKET_PLUGIN_H
#include "WebSocket.h"
#include "PnkS2t.h"
#include "plugins/AudioBuffer.h"
#include <jsoncpp/json/json.h>
namespace pnk
{
	class PnkSpeechToText;
	class WebSocketPlugin
	{
	public:
		WebSocketPlugin(PnkSpeechToText* service, int minChunkMsec) : _service(service), _minChunkMsec(minChunkMsec) {};
		void run(int port, std::string certFile, std::string keyFile, size_t maxSizeBuffer = AUDIO_BUFFER_MAX_SIZE);
		
	private:
		PnkSpeechToText* _service;
		int _minChunkMsec;
		std::unordered_map<session_id_t, ClientConnection> _connectionSessionMapping;
		std::unordered_map<session_id_t, AudioBuffer*> _bufferSessionMapping;
		w2l::streaming::LoadDataIntoSessionMethod getLoadDataFunc(int minChunkMsec, AudioBuffer *buffer);
		w2l::streaming::OutputSessionMethod getFinalOutputFunc(session_id_t sid, WebsocketServer* server);
		w2l::streaming::OutputSessionMethod getTmpOutputFunc(session_id_t sid, WebsocketServer* server);

	};
} // namespace pnk

#endif //WEBSOCKET_PLUGIN_H