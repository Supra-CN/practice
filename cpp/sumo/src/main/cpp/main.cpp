#include <iostream>
#include <list>
#include <boost/algorithm/string/join.hpp>
//#include <string>

//using namespace std;

void greeting(bool ask) {
    std::string name = "there";
    if (ask) {
        std::cout << "enter name: ";
        std::cin >> name;
        std::cout << std::endl;
    }
    std::cout << "hello, " + name
              << std::endl;
}

int finish() {
    std::cout << "bye";
    return 0;
}

void playGame() {
    std::string exit = "exit";
    int last = 2;
    int current = 3;
    int next = last + current;
    std::string answer;
    int loop = 0;
    int count = 0;
    std::list<std::string> sequence;
    sequence.push_back(std::to_string(last));
    sequence.push_back(std::to_string(current));
    while (true) {
        loop++;
        std::string joined = boost::algorithm::join(sequence, ", ");
        std::cout << std::endl
                  << "[loop = " << loop << "] (input 'exit' for quit)" << std::endl
                  << "==> input next num for sequence" << std::endl
                  << "==> " << joined << ", ? " << std::endl
                  << "==> answer: ";
        std::cin >> answer;
        if (exit == answer) {
            break;
        }
        int i_answer;
        bool ok = false;
        try {
            i_answer = std::stoi(answer);
            ok = next == i_answer;
        }
        catch (const std::invalid_argument &e) {
            std::cout << "Invalid input. Please try again! e = " << e.what() << std::endl;
        }
        if (ok) {
            count++;
            last = current;
            current = next;
            next = last + next;
            sequence.push_back(std::to_string(current));
            std::cout << "win" << std::endl;
        }
        std::cout << (ok ? "pass" : "try again")
                  << " positives = " << ((float) count / loop)
                  << std::endl;
    }
}

int main() {
    greeting(false);
    playGame();
    return finish();
}


