#include <iostream>
#include <iomanip>
#include <array>

#include <mrf/graph.h>
#include <opencv2/opencv.hpp>

template <typename T>
inline std::string
get_filled (T const& value, int width, char fill)
{
    std::stringstream ss;
    ss << std::setw(width) << std::setfill(fill) << value;
    return ss.str();
}

float potts(int, int, int l1, int l2) {
    return l1 == l2 && l1 != 0 && l2 != 0 ? 0 : 1 * MRF_MAX_ENERGYTERM;
}

int main() {
    using MRF = mrf::Graph::Ptr;

    cv::Mat3b whale = cv::imread("w_74.jpg");
    const double scale = 0.125;
    cv::resize(whale, whale, cv::Size(), scale, scale, cv::INTER_LANCZOS4);
    //cv::dilate(whale, whale, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
    cv::imshow("whale", whale);

    cv::Mat3b whale_edges(whale.size());
    cv::Laplacian(whale, whale_edges, CV_8UC3, 1);
    cv::imshow("whale_edges", whale_edges);

    cv::waitKey(1);

    // Lets start from large and detailed graph: node for each pixel
    // Labels:
    // 0 - unknown
    // 1 - water
    // 2 - espuma
    // 3 - whale
    MRF mrf = mrf::Graph::create(whale.total(), 3 + 1, mrf::LBP);

    std::cout << "Image size: " << whale.cols << "x" << whale.rows << std::endl;


    const auto to_index = [&whale](const cv::Point& p) -> int { return p.y * whale.cols + p.x;};

    {
        std::cout << "Set neighbours..." << std::flush;
        // Set neighbours
        // Each point has 2, 3 or 4 neighbours, don't care about border condition optimization
        for (auto it = whale.begin(); it != whale.end(); ++it) {
            const cv::Point p = it.pos();
            if (0 < p.x)
                mrf->set_neighbors(to_index(p), to_index({p.x - 1, p.y}));
            if (p.x < (whale.cols - 1))
                mrf->set_neighbors(to_index(p), to_index({p.x + 1, p.y}));
            if (0 < p.y)
                mrf->set_neighbors(to_index(p), to_index({p.x, p.y - 1}));
            if (p.y < (whale.rows - 1))
                mrf->set_neighbors(to_index(p), to_index({p.x, p.y + 1}));
        }
        std::cout << "Ok" << std::endl;
    }

    // Set data costs
    {
        const cv::Scalar mean = cv::mean(whale);

        std::cout << "Mean blue value: " << mean[0] / 255.f << std::endl ;
        std::cout << "Set data costs..." << std::flush;
        std::array<std::vector<mrf::SparseDataCost>, 4> costs;
        for (auto it = whale.begin(); it != whale.end(); ++it) {
            const cv::Point p = it.pos();
            const int index = to_index(p);

            costs[0].push_back({index, MRF_MAX_ENERGYTERM});

            const cv::Vec3b pixel_color = whale.at<cv::Vec3b>(p);
            const cv::Vec3f pixel_color_f(pixel_color[0]/255.f,pixel_color[1]/255.f,pixel_color[2]/255.f);
            const cv::Vec3f mean_color = cv::Vec3f(mean[0]/255.f,mean[1]/255.f,mean[2]/255.f);
            const cv::Vec3f white(1.f, 1.f, 1.f);


            const float water_prob = 7 * cv::norm(pixel_color_f, mean_color) * MRF_MAX_ENERGYTERM;
            const float espuma_prob = MRF_MAX_ENERGYTERM;//0.37 * cv::norm(pixel_color_f, white) * MRF_MAX_ENERGYTERM;
            const float whale_prob = MRF_MAX_ENERGYTERM - water_prob; //espuma_prob > water_prob ? cv::norm(pixel_color_f, mean_color) * MRF_MAX_ENERGYTERM : MRF_MAX_ENERGYTERM;
            costs[1].push_back({index, water_prob});
            costs[2].push_back({index, espuma_prob});
            costs[3].push_back({index, whale_prob});

            std::cout << /*espuma_prob << ' ' << */water_prob << ' ';
            if (p.x == (whale.cols-1)) std::cout << std::endl;
        }
        for(int i = 1; i < 4; ++i) {
            mrf->set_data_costs(i, costs[i]);
        }
        mrf->set_data_costs(0, costs[0]);

        std::cout << "Ok" << std::endl;

        mrf->set_smooth_cost(potts);
    }

    {
        // Optimization

        mrf::ENERGY_TYPE const zero = mrf::ENERGY_TYPE(0);
        mrf::ENERGY_TYPE last_energy = zero;
        mrf::ENERGY_TYPE energy = mrf->compute_energy();
        mrf::ENERGY_TYPE diff = last_energy - energy;

        unsigned int iter = 0;
        unsigned int comp = 0;
        unsigned int min_iter = 1;

        const bool verbose = 1;
        if (verbose) {
            std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
        }
        while (diff != zero) {
            if (verbose) {
                std::cout << "\t" << iter << "\t" << energy
                << "\t" << 0 /*timer.get_elapsed_sec()*/ << std::endl;
            }
            last_energy = energy;
            ++iter;
            energy = mrf->optimize(1);
            diff = last_energy - energy;
            if (diff <= zero && iter > min_iter) break;
        }

        if (verbose) {
            std::cout << "\t" << iter << "\t" << energy << std::endl;
            if (diff == zero) {
                std::cout << "\t" <<  "Converged" << std::endl;
            }
            if (diff < zero) {
                std::cout << "\t"
                << "Increase of energy - stopping optimization" << std::endl;
            }
        }
    }

    cv::Mat3b labeling(whale.size(), cv::Vec3b(0,0,0));
    {
        //std::cout << "Set neighbours..." << std::flush;
        // Set neighbours
        // Each point has 2, 3 or 4 neighbours, don't care about border condition optimization
        for (auto it = whale.begin(); it != whale.end(); ++it) {
            const cv::Point p = it.pos();

            switch (mrf->what_label(to_index(p))) {
                case 0: break;
                case 1: labeling(p) = cv::Vec3b(255,0,0); break;    // water
                case 2: labeling(p) = cv::Vec3b(255,255,255);break; // espuma
                case 3: labeling(p) = cv::Vec3b(0,255,0);break;     // whale
            }


        }
        //std::cout << "Ok" << std::endl;
    }

    cv::imshow("labeling", labeling);
    cv::waitKey(0);

    std::cout << "Done" << std::endl;

    return 0;
}