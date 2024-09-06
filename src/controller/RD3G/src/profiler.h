#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <stdexcept>

using std::cout;
using std::endl;

template<bool enabled=true>
class Profiler {
    private:
        std::map<std::string, std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>> start_time;
        std::map<std::string, std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>> end_time;
    public:
        void s(){
            if (!enabled){return;}
            s("_global");
        }
        void s(std::string label){
            if (!enabled){return;}
            auto search = start_time.find(label);
            if (search == start_time.end()){
                start_time[label] = {};
                end_time[label] = {};
                search = start_time.find(label);
            }
            search -> second.push_back(std::chrono::high_resolution_clock::now());
        }

        void e(){
            if (!enabled){return;}
            e("_global");
        }
        void e(std::string label){
            if (!enabled){return;}
            auto time = std::chrono::high_resolution_clock::now();
            auto search = end_time.find(label);
            if (search == end_time.end()){
                throw std::runtime_error("Cannot find corresponding label " + label + " did you call s() first?");
            }
            search -> second.push_back(std::move(time));
        }

        void summary(){
            if (!enabled){return;}
            std::map<std::string, double> mean_time;
            for (auto const& val : start_time){
                auto const& start = val.second;
                auto const& end = end_time[val.first];
                if (start.size() != end.size()){
                    throw std::runtime_error(" start_time and end_time do not match ");
                }
                double sum = 0.0;
                for (int i = 0; i<start.size(); i++){
                    sum += std::chrono::duration<double>(end[i] - start[i]).count();
                }
                mean_time[val.first] = sum/start.size();
            }
            double total_accounted_time = 0.0;
            double total_time = mean_time["_global"];

            cout << " ----- profile ----- " << endl;
            for (auto const& val : mean_time){
                if (val.first == "_global"){
                    continue;
                }
                total_accounted_time += val.second;
                cout << val.first << ": \t " << 100.0*(val.second/total_time) << "%" << endl;
            }
            cout << "unaccounted: \t " << 100.0*(1.0-total_accounted_time/total_time) << "%" << endl;
            cout << "overall mean time: \t " << total_time << "s" << endl;
            cout << "overall freq: \t " << 1.0/total_time << "Hz" << endl;

        }
};
