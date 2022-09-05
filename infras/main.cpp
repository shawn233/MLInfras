#include <iostream>
#include <cstdio>
#include <cmath>
#include "common.h"
#include "myresnet.h"
#include "mydataloader.h"

// #define PREDICTION_PATH "./out.txt"
#ifndef STOP_AT
#define STOP_AT -1
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 256
#endif

#ifndef POC_DIR
#define POC_DIR ".."
#endif

using namespace std;

int main(void)
{
	// legacy test code for tensor.h
	// Tensor<double> a(3, 3, 4, 5);
	// double *ptr = a.get_pointer();
	// ptr[0] = 1.5;
	// ptr[1] = 2.;
	// ptr[2] = 3.5;
	// Tensor<double> b(a);
	// Tensor<double> c;
	// c = a;
	// cout << a.at(0, 0, 0) << "\t" << b.at(0, 0, 1) << "\t" << c.at(0, 0, 2) << endl;
	// cout << a << endl;

	// Hyperparams
	const double alpha = 0.15;				// model scaling factor
	const int batch_size = BATCH_SIZE;		// batch size
	const int n_classes = 43;				// total number of classification categories
	const int stop_at = STOP_AT;			// for debug, stop after this batch

	// Define two neural networks. Then load in their parameters.
	printf("Initailizing ResNet34 task model\n");
	ResNet34 task_model(64);

	printf("Initializing ResNet34 checker model (alpha = %.2f) ...\n", alpha);
	ResNet34 checker_model((int)round(64 * alpha));
	
	const string task_params_path = POC_DIR "/model/traffic/best.task.txt";
	cout << "Loading task model parameters from " << task_params_path << endl;
	ifstream task_in(task_params_path, ios::in);
	if (!task_in)
	{
		printf("Failed to open file %s\n", task_params_path.c_str());
		exit(1);
	}
	task_model.load_from(task_in);
	task_in.close();

	const string checker_params_path = POC_DIR "/model/traffic/best.checker.txt";
	cout << "Loading checker model parameters from " << checker_params_path << endl;
	ifstream checker_in(checker_params_path, ios::in);
	if (!checker_in)
	{
		printf("Failed to open file %s\n", checker_params_path.c_str());
		exit(1);
	}
	checker_model.load_from(checker_in);
	checker_in.close();

	// Load testing dataset
	const string testset_path = POC_DIR "/data/traffic/";
	cout << "Loading testing dataset from " << testset_path << endl;
	TrafficTestLoader test_loader(testset_path, batch_size, POC_DIR "infras/dump/", false);

	// Run evaluation
	const int n_samples = test_loader.get_total_samples();
	const int n_batches = (int)ceil( (double)n_samples / batch_size );
	int total_corrects = 0;
	int total_consistents = 0;
	int running_samples = 0;

	if (stop_at >= 0)
	{
		printf(
			"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
			stop_at + 1);
	}

	#ifdef PREDICTION_PATH
	ofstream prediction_out(PREDICTION_PATH, ios::out);
	prediction_out << "Task\tChecker\tPred" << endl;
	#endif

	TypedTensor inputs;
	Tensor<int> labels;
	for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
	{
		if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
		int batch_samples = test_loader.next_batch(inputs, labels);

		PRINT_DEBUG("Forwarding in task model ...\n");
		TypedTensor task_outputs = task_model.forward(inputs);

		PRINT_DEBUG("Forwading in checker model ...\n");
		TypedTensor checker_outputs = checker_model.forward(inputs); // (B, n_classes)
		// print_vector(outputs.get_shape());

		// cout << "Inputs:\n" << inputs << endl;
		// cout << "Labels:\n" << labels << endl;
		// cout << "Outputs:\n" << outputs << endl;

		// compute correct predictions from outputs (confidence scores)
		int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
		int pred_i; 						// final predicted label for sample i
		for (int i = 0; i < batch_samples; ++ i)
		{
			task_pred_i = 0;
			for (int j = 0; j < n_classes; ++ j)
			{
				if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
				{
					task_pred_i = j;
				}
			}

			checker_pred_i = 0;
			for (int j = 0; j < n_classes; ++ j)
			{
				if (checker_outputs.at(i, j) > checker_outputs.at(i, checker_pred_i))
				{
					checker_pred_i = j;
				}
			}

			if (task_pred_i == checker_pred_i)
			{
				++ total_consistents;
			}
			pred_i = task_pred_i; // always adopt the task model prediction

			// cout << "[" << i << "] " << pred_i << endl;

			if (pred_i == labels.at(i))
			{
				++ total_corrects;
			}
			// else
			// {
			// 	printf("%d != %d\n", pred_i, labels.at(i));
			// 	getchar();
			// }

			// write prediction to file
			#ifdef PREDICTION_PATH
			prediction_out << task_pred_i << "\t" << checker_pred_i << "\t" << pred_i << endl;
			#endif
		}

		// report stats after each batch
		running_samples += batch_samples;
		printf(
			"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\tRunning consistency: %.2f%% ( %d / %d )\n", 
			batch_idx + 1, n_batches,
			(double)total_corrects * 100. / running_samples, total_corrects, running_samples,
			(double)total_consistents * 100. / running_samples, total_consistents, running_samples);
		fflush(stdout);

		if (batch_idx == stop_at) break;
	}

	#ifdef PREDICTION_PATH
	prediction_out.close();
	#endif

	// printf(
	//     "[Final] Prediction accuracy: %.2f%% ( %d / %d )\tRunning consistency: %.2f%% ( %d / %d)\n", 
	//     (double)total_corrects * 100. / n_samples, total_corrects, n_samples,
	// 	(double)total_consistents * 100. / running_samples, total_consistents, running_samples);
	printf(
		"*%.4f%%\t%.4f%%\n", 
		(double)total_corrects * 100. / running_samples, 
		(double)total_consistents * 100. / running_samples);

	return 0;
}