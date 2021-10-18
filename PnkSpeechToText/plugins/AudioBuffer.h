/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef AUDIO_BUFFER_H
#define AUDIO_BUFFER_H
#include <stdio.h>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <glog/logging.h> //logging
namespace pnk
{
#define AUDIO_BUFFER_MAX_SIZE 65536
    class AudioBuffer
    {
    public:
        AudioBuffer(size_t maxSize) : _maxSize(maxSize), _sizeInBytes(0)
        {
            data = new char[_maxSize];
            head = tail = data;
            _isFull = false;
            _isEnd = false;
            _tailOffset = _maxSize;
            _headOffset = _maxSize;
        };
        ~AudioBuffer()
        {
            delete[] data;
        }
        size_t pull(char *dst, size_t size)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            while ((_sizeInBytes < size) && (!_isEnd))
            {
                cond_.wait(mlock);
            }
            if (size > _sizeInBytes)
                size = _sizeInBytes;
            if (_headOffset < size)
            {
                memcpy(dst, head, _headOffset);
                memcpy(dst + _headOffset, data, size - _headOffset);
                head = data + (size - _headOffset);
                _headOffset = _maxSize - (size - _headOffset);
            }
            else
            {
                memcpy(dst, head, size);
                head += size;
                _headOffset -= size;
            }
            _isFull = false;
            _sizeInBytes -= size;
            return size;
        }
        void push(const char *src, size_t size)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            if (_tailOffset < size)
            {
                memcpy(tail, src, _tailOffset);
                memcpy(data, src + _tailOffset, size - _tailOffset);
                tail = data + (size - _tailOffset);
                _tailOffset = _maxSize - (size - _tailOffset);
                if (_sizeInBytes + size > _maxSize)
                {
                    LOG(WARNING) << "Audio buffer is full, drop latest packets";
                    _isFull = true;
                    head = tail;
                    _headOffset = _tailOffset;
                    _sizeInBytes = _maxSize;
                }
                else
                {
                    _sizeInBytes += size;
                }
            }
            else
            {
                memcpy(tail, src, size);
                tail += size;
                _tailOffset -= size;
                if (_sizeInBytes + size > _maxSize)
                {
                    LOG(WARNING) << "Audio buffer is full, drop latest packets";
                    _isFull = true;
                    head = tail;
                    _headOffset = _tailOffset;
                    _sizeInBytes = _maxSize;
                }
                else
                {
                    _sizeInBytes += size;
                }
            }
            mlock.unlock();
            cond_.notify_all();
        }
        size_t size()
        {
            return _sizeInBytes;
        }
        bool isEmpty()
        {
            return (_sizeInBytes == 0);
        }
        bool isFull()
        {
            return _isFull;
        }
        void dumpDataI16()
        {
            printf("Audio buffer %p (sizeInBytes = %zu, head = %p, tail = %p, headOffset = %zu, tailOffset = %zu) data: [", data, _sizeInBytes, head, tail, _headOffset, _tailOffset);
            const int16_t *d = (int16_t *)data;
            for (int i = 0; i < _maxSize / sizeof(int16_t); i++)
            {
                printf("%d ", d[i]);
            }
            printf("]\n");
        }
        void destroy()
        {
            _isEnd = true;
            cond_.notify_all();
        }

    private:
        char *data;
        char *head;
        char *tail;
        size_t _tailOffset;
        size_t _headOffset;
        std::mutex mutex_;
        std::condition_variable cond_;
        size_t _sizeInBytes;
        size_t _maxSize;
        bool _isFull;
        bool _isEnd;
    };
} // namespace pnk
#endif //AUDIO_BUFFER_H