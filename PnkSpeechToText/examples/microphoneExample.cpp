/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PnkS2t.h"
#include <iostream>
void help() {
    std::cout << "Usage: ./microphoneExample config_file.cfg device sample_rate channels\n";
}
int main(int argc, char* argv[]) {
    if ((argc != 5))
    {
        std::cout << "Invalid arguments\n";
        help();
        return 0;
    }
    pnk::PnkSpeechToText* manager = new pnk::PnkSpeechToText(argv[1]);
	pnk::MicroPlugin* devPlugin = manager->getDevicePlugin();
    devPlugin->run(argv[2], std::stoi(argv[3]), std::stoi(argv[4]));
}