/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "WebSocketPlugin.h"
namespace pnk
{
    w2l::streaming::LoadDataIntoSessionMethod WebSocketPlugin::getLoadDataFunc(int minChunkMsec, AudioBuffer *buffer)
    {
        return [minChunkMsec, buffer, this](std::shared_ptr<w2l::streaming::IOBuffer> input) {
            constexpr const float kMaxUint16 = static_cast<float>(0x8000);
            constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
            const int minChunkSize = minChunkMsec * kAudioWavSamplingFrequency / 1000;
            size_t minChunkSizeInBytes = minChunkSize * sizeof(float);
            auto tmpBuffer = std::make_shared<w2l::streaming::IOBuffer>(minChunkSizeInBytes);
            tmpBuffer->ensure<char>(minChunkSizeInBytes);
            char *tmpCharPtr = tmpBuffer->data<char>();
            size_t bytesRead = buffer->pull(tmpCharPtr, minChunkSizeInBytes);
            tmpBuffer->move<char>(minChunkSizeInBytes);
            int16_t *tmpPtr = tmpBuffer->data<int16_t>();
            const int tmpSize = tmpBuffer->size<int16_t>();
            input->ensure<float>(tmpSize);
            float *inputPtr = input->data<float>();
            std::transform(tmpPtr, tmpPtr + tmpSize, inputPtr, [](int16_t i) -> float {
                return static_cast<float>(i) / kMaxUint16;
            });
            if (bytesRead % sizeof(int16_t))
            {
                std::cerr << "readRtpStreamIntoBuffer(buffer=" << input->debugString()
                          << " ,sizeInBufferType=" << minChunkSizeInBytes << ") read "
                          << bytesRead << " bytes that is not devisible by "
                          << sizeof(int16_t) << std::endl;
            }
            input->move<float>(tmpSize);
            return tmpSize;
        };
    }
    w2l::streaming::OutputSessionMethod WebSocketPlugin::getFinalOutputFunc(session_id_t sid, WebsocketServer *server)
    {
        return [sid, server, this](std::string text) {
            if (_connectionSessionMapping.find(sid) != _connectionSessionMapping.end())
            {
                Json::StreamWriterBuilder builder;
                ClientConnection conn = _connectionSessionMapping[sid];
                Json::Value val;
                val["text"] = text;
                val["final"] = true;
                server->sendMessage(conn, Json::writeString(builder, val));
            }
        };
    }
    w2l::streaming::OutputSessionMethod WebSocketPlugin::getTmpOutputFunc(session_id_t sid, WebsocketServer *server)
    {
        return [sid, server, this](std::string text) {
            if (_connectionSessionMapping.find(sid) != _connectionSessionMapping.end())
            {
                Json::StreamWriterBuilder builder;
                ClientConnection conn = _connectionSessionMapping[sid];
                Json::Value val;
                val["text"] = text;
                val["final"] = false;
                server->sendMessage(conn, Json::writeString(builder, val));
            }
        };
    }
    void WebSocketPlugin::run(int port, std::string certFile, std::string keyFile, size_t maxSizeBuffer)
    {
        // Register our network callbacks, ensuring the logic is run on the main
        // thread's event loop
        asio::io_service eventLoop;
        WebsocketServer server(certFile, keyFile);
        server.connect([&eventLoop, &server, maxSizeBuffer, this](ClientConnection conn) {
            eventLoop.post([conn, &server, maxSizeBuffer, this]() {
                LOG(INFO) << "Connection opened.";
                LOG(INFO) << "There are now " << server.numConnections()
                          << " open connections.";
                session_id_t sid = _service->createSession(ASYNC);
                _connectionSessionMapping.insert({sid, conn});
                AudioBuffer *buffer = new AudioBuffer(maxSizeBuffer);
                _bufferSessionMapping[sid] = buffer;
                _service->sessionRun(sid, getLoadDataFunc(_minChunkMsec, buffer), getFinalOutputFunc(sid, &server), getTmpOutputFunc(sid, &server));
            });
        });
        server.disconnect([&eventLoop, &server, this](ClientConnection conn) {
            eventLoop.post([conn, &server, this]() {
                LOG(INFO) << "Connection closed.";
                LOG(INFO) << "There are now " << server.numConnections()
                          << " open connections.";
                for (auto &it : _connectionSessionMapping)
                    if (it.second.lock().get() == conn.lock().get())
                    {
                        session_id_t sid = it.first;
                        AudioBuffer *buffer = _bufferSessionMapping[sid];
                        buffer->destroy();
                        _service->sessionEnd(sid);
                        _connectionSessionMapping.erase(sid);
                        _bufferSessionMapping.erase(sid);
                        delete buffer;
                        return;
                    }
                LOG(WARNING) << "Connection is not found to disconnecting";
            });
        });
        server.message(
            [&eventLoop, &server, this](ClientConnection conn, WebsocketEndpoint::message_ptr msg) {
                eventLoop.post([conn, &server, msg, this]() {
                    for (auto &it : _connectionSessionMapping)
                        if (it.second.lock().get() == conn.lock().get())
                        {
                            session_id_t sid = it.first;
                            AudioBuffer *buffer = _bufferSessionMapping[sid];
                            buffer->push(msg->get_payload().c_str(), msg->get_payload().size());
                            return;
                        }
                    LOG(WARNING) << "Receive messages from unknown connection: " << conn.lock().get();
                });
            });
        std::thread serverThread([&server, port]() { server.run(port); });
        asio::io_service::work work(eventLoop);
        eventLoop.run();
    }
} // namespace pnk