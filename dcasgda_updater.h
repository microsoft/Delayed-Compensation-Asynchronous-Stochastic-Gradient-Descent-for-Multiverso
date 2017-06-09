//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// Delayed Compensation Async Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//
// See https://arxiv.org/abs/1609.08326 for the details.
//
#ifndef MULTIVERSO_UPDATER_DCASGDA_UPDATER_H_
#define MULTIVERSO_UPDATER_DCASGDA_UPDATER_H_

#include "multiverso/updater/updater.h"

#include <vector>
#include <cmath>

namespace multiverso {

	template <typename T>
	class DCASGDAUpdater : public Updater<T> {
	public:
		explicit DCASGDAUpdater(size_t size) :
			size_(size){
			Log::Debug("[DC-ASGD-A-Updater] Init. \n");
			shadow_copies_.resize(MV_NumWorkers(), std::vector<T>(size_));
			mean_square_.resize(MV_NumWorkers(), std::vector<T>(size_, 0.));
		}

		void Update(size_t num_element, T*data, T*delta,
			AddOption* option, size_t offset) override {
			float e = 1e-7;
			for (size_t index = 0; index < num_element; ++index) {
				T g = delta[index] / option->learning_rate();

				mean_square_[option->worker_id()][index + offset] *= option->momentum();
				mean_square_[option->worker_id()][index + offset] += (1 - option->momentum()) * g * g;
                data[index + offset] -= option->learning_rate() *
					(g + option->lambda() / sqrt(mean_square_[option->worker_id()][index + offset] + e)*
					g * g *
					(data[index + offset] - shadow_copies_[option->worker_id()][index + offset]));
					
				// caching each worker's latest version of parameter
				shadow_copies_[option->worker_id()][index + offset] = data[index + offset];
			}
		}

		~DCASGDUpdater(){
			shadow_copies_.clear();
			mean_square_.clear();
		}

	protected:
		std::vector< std::vector<T>> shadow_copies_;
		std::vector< std::vector<T>> mean_square_;
		size_t size_;
	};
}

#endif // MULTIVERSO_UPDATER_DCASGDA_UPDATER_H_
