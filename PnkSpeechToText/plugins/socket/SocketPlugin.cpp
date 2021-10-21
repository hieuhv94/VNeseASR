#include "SocketPlugin.h"

namespace pnk
{
    w2l::streaming::LoadDataIntoSessionMethod SocketPlugin::getLoadDataFunc(int minChunkMsec, AudioBuffer *buffer)
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
            std::transform(tmpPtr, tmpPtr + tmpSize, inputPtr, [](int16_t i) -> float { return static_cast<float>(i) / kMaxUint16; });
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
    w2l::streaming::OutputSessionMethod SocketPlugin::getTmpOutputFunc(int client, uint8_t id)
    {
        return [client, id, this](std::string text) {
            TextChunk chunk;
            chunk.id = id;
            chunk.size = (uint8_t)text.size();
            memcpy(&chunk.data[0], text.c_str(), text.size());
            if (send(client, (void *)&chunk, sizeof(TextChunk), 0) < 0)
                LOG(ERROR) << "Failed to send text to client " << client;
        };
    }
    void SocketPlugin::run(const char *host, int port, size_t maxSizeBuffer)
    {
        int serverFd, client, valread;
        struct sockaddr_in address;
        int opt = 1;
        int addrlen = sizeof(address);
        // Creating socket file descriptor
        if ((serverFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        {
            perror("socket failed");
            exit(EXIT_FAILURE);
        }

        // Forcefully attaching socket to the port 8080
        if (setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                       &opt, sizeof(opt)))
        {
            perror("setsockopt");
            exit(EXIT_FAILURE);
        }
        address.sin_family = AF_INET;
        inet_aton(host, &address.sin_addr);
        address.sin_port = htons(port);

        // Forcefully attaching socket to the port 8080
        if (bind(serverFd, (struct sockaddr *)&address,
                 sizeof(address)) < 0)
        {
            perror("bind failed");
            exit(EXIT_FAILURE);
        }
        if (listen(serverFd, 3) < 0)
        {
            perror("listen");
            exit(EXIT_FAILURE);
        }
        while ((client = accept(serverFd, (struct sockaddr *)&address,
                                (socklen_t *)&addrlen)) >= 0)
        {
            LOG(INFO) << "Receive connection from " << client;
            AudioBuffer buffer(100 * (DATA_SIZE + HEADER_SIZE));
            ConnectionDescription connDes;
            bool isConnect = true;
            std::thread handleData([maxSizeBuffer, client, &buffer, &connDes, &isConnect, this]() {
                char tmpBuffer[DATA_SIZE + HEADER_SIZE];
                while (isConnect)
                {
                    size_t bytesRead = buffer.pull(&tmpBuffer[0], DATA_SIZE + HEADER_SIZE);
                    if (bytesRead = DATA_SIZE + HEADER_SIZE)
                    {
                        uint8_t id = tmpBuffer[0];
                        if (connDes.connectionSessionMapping.find(id) == connDes.connectionSessionMapping.end())
                        {
                            session_id_t sid = _service->createSession(ASYNC);
                            connDes.connectionSessionMapping.insert({id, sid});
                            AudioBuffer *buffer = new AudioBuffer(maxSizeBuffer);
                            connDes.bufferSessionMapping[id] = buffer;
                            _service->sessionRun(
                                sid, getLoadDataFunc(_minChunkMsec, buffer), ([](std::string text) { /*Don't anything*/ }),
                                getTmpOutputFunc(client, id));
                        }
                        connDes.bufferSessionMapping[id]->push(&tmpBuffer[1], bytesRead - 1);
                        bytesRead = 0;
                    }
                }
            });
            std::thread readData([&client, &buffer, &connDes, &isConnect, this]() {
                int bytesRead = 0;
                char tmpBuffer[DATA_SIZE + HEADER_SIZE];
                while ((bytesRead = read(client, tmpBuffer, DATA_SIZE + HEADER_SIZE)) > 0)
                {

                    buffer.push(&tmpBuffer[0], bytesRead);
                }
                LOG(INFO) << "Disconnect from " << client;
                isConnect = false;
                buffer.destroy();
                for (auto buff : connDes.bufferSessionMapping)
                {
                    buff.second->destroy();
                }
                for (auto sess : connDes.connectionSessionMapping)
                {
                    _service->sessionEnd(sess.second);
                }
                for (auto buff : connDes.bufferSessionMapping)
                {
                    delete buff.second;
                }
            });

            handleData.join();
            readData.join();
        }
    }

} // namespace pnk
