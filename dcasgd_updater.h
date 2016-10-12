//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// Delayed Compensation Async Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//
#ifndef MULTIVERSO_UPDATER_DCASGD_UPDATER_H_
#define MULTIVERSO_UPDATER_DCASGD_UPDATER_H_

#include "updater.h"

#include "vector"
#include "cmath"

namespace multiverso {

	template <typename T>
	class DCASGDUpdater : public Updater<T> {
	public:
		explicit DCASGDUpdater(size_t size, bool isPipeline) :
			size_(size){
			Log::Debug("[DC-ASGDUpdater] Init. \n");
			shadow_copies_.resize(isPipeline ? MV_NumWorkers() * 2 : MV_NumWorkers(), std::vector<T>(size_));
			smooth_gradient_.resize(size_);
		}

		void Update(size_t num_element, T*data, T*delta,
			AddOption* option, size_t offset) override {
			for (size_t index = 0; index < num_element; ++index) {
				//if (index == 0 && offset == 0)
				//	Log::Info("da:%f, co:%f, de:%f, lr:%f, lam:%f, ",
				//	data[index + offset], shadow_copies_[option->worker_id()][index + offset], delta[index],
				//	option->learning_rate(), option->lambda());

				data[index + offset] -= option->learning_rate() *
					(delta[index] / option->learning_rate() - option->lambda() *
					std::abs(delta[index] / option->learning_rate()) *
					(data[index + offset] - shadow_copies_[option->worker_id()][index + offset]));

				// caching each worker's latest version of parameter
				shadow_copies_[option->worker_id()][index + offset] = data[index + offset];
				//if (index == 0 && offset == 0)
				//	fprintf(stderr, "nda:%f, nco:%f\n", data[index + offset], shadow_copies_[option->worker_id()][index + offset]);
				//if (index == 0 && offset == 0)
				//	Log::Debug("nda:%f, nco:%f\n", data[index + offset], shadow_copies_[option->worker_id()][index + offset]);
			}
		}

		~DCASGDUpdater(){
			shadow_copies_.clear();
			smooth_gradient_.clear();
		}

	protected:
		std::vector< std::vector<T>> shadow_copies_;
		std::vector<T> smooth_gradient_;
		size_t size_;
	};
}

#endif // MULTIVERSO_UPDATER_DCASGD_UPDATER_H_
