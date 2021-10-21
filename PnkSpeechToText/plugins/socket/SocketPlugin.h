/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef SOCKET_PLUGIN_H
#define SOCKET_PLUGIN_H
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unordered_map>
#include "plugins/AudioBuffer.h"
#include "PnkS2t.h"
#define DATA_SIZE 2048
#define HEADER_SIZE 1
namespace pnk
{
    class PnkSpeechToText;
    struct ConnectionDescription
    {
        std::unordered_map<uint8_t, session_id_t> connectionSessionMapping;
        std::unordered_map<uint8_t, AudioBuffer *> bufferSessionMapping;
    };
    struct TextChunk
    {
        uint8_t id;
        uint8_t size;
        char data[50] = {0};
    };
    class SocketPlugin
    {
    public:
        SocketPlugin(PnkSpeechToText *service, int minChunkMsec) : _service(service), _minChunkMsec(minChunkMsec){};
        void run(const char *host, int port, size_t maxSizeBuffer = AUDIO_BUFFER_MAX_SIZE);

    private:
        PnkSpeechToText *_service;
        int _minChunkMsec;
        w2l::streaming::LoadDataIntoSessionMethod getLoadDataFunc(int minChunkMsec, AudioBuffer *buffer);
        w2l::streaming::OutputSessionMethod getTmpOutputFunc(int client, uint8_t id);
    };
}
#endif //SOCKET_PLUGIN_H