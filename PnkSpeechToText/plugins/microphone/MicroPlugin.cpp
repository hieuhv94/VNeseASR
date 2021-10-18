/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "MicroPlugin.h"
namespace pnk
{
    w2l::streaming::LoadDataIntoSessionMethod MicroPlugin::getLoadDataFunc(int minChunkMsec)
    {
        int fd = open("in.pcm", O_CREAT | O_RDWR, 0666);

        return [minChunkMsec, fd, this](std::shared_ptr<w2l::streaming::IOBuffer> input) {
            constexpr const float kMaxUint16 = static_cast<float>(0x8000);
            constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
            const int minChunkSize = minChunkMsec * kAudioWavSamplingFrequency / 1000;
            size_t minChunkSizeInBytes = minChunkSize * sizeof(float);
            auto tmpBuffer = std::make_shared<w2l::streaming::IOBuffer>(minChunkSizeInBytes);
            tmpBuffer->ensure<char>(minChunkSizeInBytes);
            char *tmpCharPtr = tmpBuffer->data<char>();
            size_t bytesRead = _buffer.pull(tmpCharPtr, minChunkSizeInBytes);
            write(fd, tmpCharPtr, bytesRead);
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
    w2l::streaming::OutputSessionMethod MicroPlugin::getFinalOutputFunc()
    {
        return [](std::string text) {
            std::cout << text << std::endl;
        };
    }
    void MicroPlugin::run(const char *device, unsigned int sampleRate, int channels)
    {
        session_id_t sid = _service->createSession(ASYNC);
        _service->sessionRun(
            sid, getLoadDataFunc(_minChunkMsec),getFinalOutputFunc(),[](std::string text) {
            // Don't anything
            });

        // Read audio buffer from sound device
        int rc;
        int size;
        snd_pcm_t *handle;
        snd_pcm_hw_params_t *params;
        int dir;
        snd_pcm_uframes_t frames = 2048;
        char *buffer;

        /* Open PCM device for recording (capture). */
        rc = snd_pcm_open(&handle, device,
                          SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0)
        {
            LOG(ERROR) << "Unable to open pcm device: " << snd_strerror(rc);
            exit(1);
        }

        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        snd_pcm_hw_params_any(handle, params);

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        snd_pcm_hw_params_set_access(handle, params,
                                     SND_PCM_ACCESS_RW_INTERLEAVED);

        /* Signed 16-bit little-endian format */
        snd_pcm_hw_params_set_format(handle, params,
                                     SND_PCM_FORMAT_S16_LE);

        /* Set channels */
        snd_pcm_hw_params_set_channels(handle, params, channels);

        /* Set sample rate */
        snd_pcm_hw_params_set_rate_near(handle, params, &sampleRate, &dir);

        snd_pcm_hw_params_set_period_size_near(handle,
                                               params, &frames, &dir);

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(handle, params);
        if (rc < 0)
        {
            LOG(ERROR) << "Unable to set hw parameters: " << snd_strerror(rc);
            exit(1);
        }

        /* Use a buffer large enough to hold one period */
        snd_pcm_hw_params_get_period_size(params,
                                          &frames, &dir);
        size = frames * channels * 2; /* 2 bytes/sample, 2 channels */
        buffer = (char *)malloc(size);
        ReSampler resampler(channels, 1, (channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO, AV_CH_LAYOUT_MONO, sampleRate, 16000, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16);
        while ((rc = snd_pcm_readi(handle, buffer, frames)) == frames)
        {
            int framesCount = resampler.run((uint8_t **)&buffer, frames, (const uint8_t **)&buffer, frames);
            _buffer.push(buffer, framesCount * 2);
        }
        if (rc == -EPIPE)
        {
            /* EPIPE means overrun */
            LOG(ERROR) << "Qverrun occurred " << snd_pcm_prepare(handle);
        }
        else if (rc < 0)
        {
            LOG(ERROR) << "Error from read: " << snd_strerror(rc);
        }
        else if (rc != (int)frames)
        {
            LOG(ERROR) << "Short read, read %d frames" << rc;
        }
        snd_pcm_drain(handle);
        snd_pcm_close(handle);
        free(buffer);
    }
} // namespace pnk