//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// Delayed Compensation Async Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//
// See https://arxiv.org/abs/1609.08326 for the details.
//
#ifndef MULTIVERSO_UPDATER_DCASGD_UPDATER_H_
#define MULTIVERSO_UPDATER_DCASGD_UPDATER_H_

#include "multiverso/updater/updater.h"

#include <vector>
#include <cmath>

namespace multiverso {

    template <typename T>
    class DCASGDUpdater : public Updater<T> {
    public:
        explicit DCASGDUpdater(size_t size, bool isPipeline) :
            size_(size) {
            Log::Debug("[DC-ASGDUpdater] Init. \n");
            shadow_copies_.resize(isPipeline ? MV_NumWorkers() * 2 : MV_NumWorkers(), std::vector<T>(size_));
            mean_square_.resize(MV_NumWorkers(), std::vector<T>(size_));
            for (int i = 0; i < MV_NumWorkers(); ++i)
            {
                for (int j = 0; j < size_; ++j)
                {
                    mean_square_[i][j] = 0.;
                }
            }
        }

        void Update(size_t num_element, T* data, T* delta,
            AddOption* option, size_t offset) override {
            float e = 1e-10;
            //fprintf(stderr, "data:%f, lr:%f, delta:%f, lambda:%f, ms:%f, sc:%f, ", data[0], option->learning_rate(), delta[0], option->lambda(), mean_square_[option->worker_id()][0], shadow_copies_[option->worker_id()][0]);
            for (size_t index = 0; index < num_element; ++index) {
                T g = delta[index] / option->learning_rate();

                /******************************ASGD*********************************/
                //data[index + offset] -= option->learning_rate() * g;

                /******************************DC-ASGD-c*********************************/
                //data[index + offset] -= option->learning_rate() *
                //	(g + option->lambda() *	g * g *
                //	(data[index + offset] - shadow_copies_[option->worker_id()][index + offset]));


                /******************************DC-ASGD-a*********************************/
                mean_square_[option->worker_id()][index + offset] *= option->momentum();
                mean_square_[option->worker_id()][index + offset] += (1 - option->momentum()) * g * g;
                data[index + offset] -= option->learning_rate() *
                    (g + option->lambda() / sqrt(mean_square_[option->worker_id()][index + offset] + e)*
                        g * g *
                        (data[index + offset] - shadow_copies_[option->worker_id()][index + offset]));

                ///******************************ASGD-dev*********************************/
                //data[index + offset] -= option->learning_rate() *
                //	(g + option->lambda() *	std::abs(g) * g * g);


                //caching each worker's latest version of parameter
                shadow_copies_[option->worker_id()][index + offset] = data[index + offset];
            }
            //fprintf(stderr, "data:%f, ms:%f, sc:%f\n", data[0], mean_square_[option->worker_id()][0], shadow_copies_[option->worker_id()][0]);
            //for (size_t index = 0; index < num_element; ++index) {
            //	data[index + offset] -= delta[index];
            //}
        }

        void Access(size_t num_element, T* data, T* blob_data,
            size_t offset, AddOption*) override {
            memcpy(blob_data, data + offset, sizeof(T) * num_element);
        }

        ~DCASGDUpdater() {
            shadow_copies_.clear();
            mean_square_.clear();
        }

    protected:
        std::vector< std::vector<T>> shadow_copies_;
        std::vector< std::vector<T>> mean_square_;

        size_t size_;
    };
}

#endif // MULTIVERSO_UPDATER_DCASGD_UPDATER_H_
