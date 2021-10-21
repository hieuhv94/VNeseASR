/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PnkS2t.h"
#include <iostream>
void help() {
    std::cout << "Usage: ./socketExample config_file.cfg host port\n";
}
int main(int argc, char* argv[]) {
    if ((argc != 4))
    {
        std::cout << "Invalid arguments\n";
        help();
        return 0;
    }
    pnk::PnkSpeechToText* manager = new pnk::PnkSpeechToText(argv[1]);
	pnk::SocketPlugin* sPlugin = manager->getSocketPlugin();
    sPlugin->run(argv[2], std::stoi(argv[3]));
}