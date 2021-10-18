/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef FILE_PLUGIN_H
#define FILE_PLUGIN_H
#include <fstream>
#include <memory>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include "PnkS2t.h"
using namespace w2l;
using namespace streaming;
namespace pnk
{

    class PnkSpeechToText;
    class FilePlugin
    {
    public:
        FilePlugin(PnkSpeechToText *service, int minChunkMsec) : _service(service), _minChunkMsec(minChunkMsec) {};
        void runSync(const char *audioPath, std::string &output);
        void runInteractiveMode();
        
    private:
        PnkSpeechToText* _service;
        int _minChunkMsec;
        w2l::streaming::LoadDataIntoSessionMethod getLoadDataFunc(std::ifstream &inputAudioStream,
                                                                  int minChunkMsec);
        w2l::streaming::OutputSessionMethod getOutputFunc(std::string &output);

        int readStreamIntoBuffer(
            std::istream &inputStream,
            std::shared_ptr<IOBuffer> buffer,
            int bytesToRead)
        {
            assert(bytesToRead > 0);
            assert(buffer);
            int bytesRead = 0;
            buffer->ensure<char>(bytesToRead);
            char *inputPtrChar = buffer->data<char>();
            while (bytesRead < bytesToRead && inputStream.good())
            {
                inputStream.read(inputPtrChar + bytesRead, bytesToRead - bytesRead);
                bytesRead += inputStream.gcount();
            }
            buffer->move<char>(bytesRead);
            return bytesRead;
        }
        // Read type StreamType off inputStream, apply transformationFunction to each
        // element and write the tranformed elemnt into buffer. Retutns the number os
        // BufferType elements written into buffer.
        template <typename StreamType, typename BufferType>
        int readTransformStreamIntoBuffer(
            std::istream &inputStream,
            std::shared_ptr<IOBuffer> buffer,
            int sizeInBufferType,
            const std::function<BufferType(StreamType)> &transformationFunction)
        {
            const int sizeInBytes = sizeInBufferType * sizeof(BufferType);
            auto tmpBuffer = std::make_shared<IOBuffer>(sizeInBytes);
            const int bytesRead =
                readStreamIntoBuffer(inputStream, tmpBuffer, sizeInBytes);
            int16_t *tmpPtr = tmpBuffer->data<int16_t>();
            const int tmpSize = tmpBuffer->size<int16_t>();
            buffer->ensure<float>(tmpSize);
            float *bufferPtr = buffer->data<float>();
            std::transform(tmpPtr, tmpPtr + tmpSize, bufferPtr, transformationFunction);
            if (bytesRead % sizeof(StreamType))
            {
                std::cerr << "readTransformIntoBuffer(buffer=" << buffer->debugString()
                          << " ,sizeInBufferType=" << sizeInBufferType << ") read "
                          << bytesRead << " bytes that is not devisible by "
                          << sizeof(StreamType) << std::endl;
            }
            buffer->move<float>(tmpSize);
            return tmpSize;
        }
    };
} // namespace pnk
#endif //FILE_PLUGIN_H