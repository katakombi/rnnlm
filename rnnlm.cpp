///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3g
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
// (c) 2013      Stefan Kombrink (katakombi@gmail.com)
//
///////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "rnnlmlib.h"

using namespace std;

int argPos(char *str, int argc, char **argv)
{
    int a;

    for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;

    return -1;
}

int main(int argc, char **argv)
{
    int i;

    int debug_mode=1;

    int fileformat=TEXT;

    int train_mode=0;
    int valid_data_set=0;
    int test_data_set=0;
    int rnnlm_file_set=0;

    int alpha_set=0, train_file_set=0;

    int class_size=100;
    int old_classes=0;
    float lambda=0.75;
    float gradient_cutoff=15;
    float dynamic=0;
    float starting_alpha=0.1;
    float regularization=0.0000001;
    float min_improvement=1.003;
    int hidden_size=30;
    int compression_size=0;
    long long direct=0;
    int direct_order=3;
    int bptt=0;
    int bptt_block=10;
    int gen=0;
    int independent=0;
    int use_lmprob=0;
    int rand_seed=1;
    int nbest=0;
    int one_iter=0;
    int anti_k=0;
    int ncluster=0;
    int kmean_iter=-1;

    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    char compress_file[MAX_STRING];
    char lmprob_file[MAX_STRING];

    FILE *f;

    compress_file[0]=0;

    if (argc==1) {
    	//printf("Help\n");

    	fprintf(stdout,"Recurrent neural network based language modeling toolkit v 0.3g\n\n");

    	fprintf(stdout,"Options:\n");

    	//
    	fprintf(stdout,"Parameters for training phase:\n");

    	fprintf(stdout,"\t-train <file>\n");
        fprintf(stdout,"\t\tUse text data from <file> to train rnnlm model\n");

        fprintf(stdout,"\t-class <int>\n");
        fprintf(stdout,"\t\tWill use specified amount of classes to decompose vocabulary; default is 100\n");

	fprintf(stdout,"\t-old-classes\n");
        fprintf(stdout,"\t\tThis will use old algorithm to compute classes, which results in slower models but can be a bit more precise\n");

    	fprintf(stdout,"\t-rnnlm <file>\n");
        fprintf(stdout,"\t\tUse <file> to store rnnlm model\n");

        fprintf(stdout,"\t-binary\n");
        fprintf(stdout,"\t\tRnnlm model will be saved in binary format (default is plain text)\n");

    	fprintf(stdout,"\t-valid <file>\n");
    	fprintf(stdout,"\t\tUse <file> as validation data\n");

    	fprintf(stdout,"\t-alpha <float>\n");
    	fprintf(stdout,"\t\tSet starting learning rate; default is 0.1\n");

    	fprintf(stdout,"\t-beta <float>\n");
    	fprintf(stdout,"\t\tSet L2 regularization parameter; default is 1e-7\n");

    	fprintf(stdout,"\t-hidden <int>\n");
    	fprintf(stdout,"\t\tSet size of hidden layer; default is 30\n");

    	fprintf(stdout,"\t-compression <int>\n");
    	fprintf(stdout,"\t\tSet size of compression layer; default is 0 (not used)\n");

    	fprintf(stdout,"\t-direct <int>\n");
    	fprintf(stdout,"\t\tSets size of the hash for direct connections with n-gram features in millions; default is 0\n");

    	fprintf(stdout,"\t-direct-order <int>\n");
    	fprintf(stdout,"\t\tSets the n-gram order for direct connections (max %d); default is 3\n", MAX_NGRAM_ORDER);

    	fprintf(stdout,"\t-bptt <int>\n");
    	fprintf(stdout,"\t\tSet amount of steps to propagate error back in time; default is 0 (equal to simple RNN)\n");

    	fprintf(stdout,"\t-bptt-block <int>\n");
    	fprintf(stdout,"\t\tSpecifies amount of time steps after which the error is backpropagated through time in block mode (default 10, update at each time step = 1)\n");

    	fprintf(stdout,"\t-one-iter\n");
    	fprintf(stdout,"\t\tWill cause training to perform exactly one iteration over training data (useful for adapting final models on different data etc.)\n");

    	fprintf(stdout,"\t-anti-kasparek <int>\n");
    	fprintf(stdout,"\t\tModel will be saved during training after processing specified amount of words\n");

    	fprintf(stdout,"\t-min-improvement <float>\n");
    	fprintf(stdout,"\t\tSet minimal relative entropy improvement for training convergence; default is 1.003\n");

    	fprintf(stdout,"\t-gradient-cutoff <float>\n");
    	fprintf(stdout,"\t\tSet maximal absolute gradient value (to improve training stability, use lower values; default is 15, to turn off use 0)\n");

        fprintf(stdout,"\t-rand-seed <int>\n");
        fprintf(stdout,"\t\tSet the initialization value for the random number generator; use this to train complementary models\n");

    	//
    	fprintf(stdout,"Parameters for testing phase:\n");

    	fprintf(stdout,"\t-rnnlm <file>\n");
    	fprintf(stdout,"\t\tRead rnnlm model from <file>\n");

    	fprintf(stdout,"\t-test <file>\n");
    	fprintf(stdout,"\t\tUse <file> as test data to report perplexity\n");

    	fprintf(stdout,"\t-lm-prob\n");
    	fprintf(stdout,"\t\tUse other LM probabilities for linear interpolation with rnnlm model; see examples at the rnnlm webpage\n");

    	fprintf(stdout,"\t-lambda <float>\n");
    	fprintf(stdout,"\t\tSet parameter for linear interpolation of rnnlm and other lm; default weight of rnnlm is 0.75\n");

    	fprintf(stdout,"\t-dynamic <float>\n");
    	fprintf(stdout,"\t\tSet learning rate for dynamic model updates during testing phase; default is 0 (static model)\n");



        fprintf(stdout,"\t-compress <int>\n");
        fprintf(stdout,"\t\tCompress the ME part of an RNNME model (direct connections)\n");
        fprintf(stdout,"\t\tUse given number of bits for compression (between 3 and 8)\n");

        fprintf(stdout,"\t-kmean <int>\n");
        fprintf(stdout,"\t\tImprove compression by given number of iterations of kmean clustering\n");

        fprintf(stdout,"\t-write-compressed <file>\n");
        fprintf(stdout,"\t\tWrite the compressed model to disk\n");

    	//

    	fprintf(stdout,"Additional parameters:\n");

    	fprintf(stdout,"\t-gen <int>\n");
    	fprintf(stdout,"\t\tGenerate specified amount of words given distribution from current model\n");

    	fprintf(stdout,"\t-independent\n");
    	fprintf(stdout,"\t\tWill erase history at end of each sentence (if used for training, this switch should be used also for testing & rescoring)\n");

    	fprintf(stdout,"\nExamples:\n");
    	fprintf(stdout,"rnnlm -train train -rnnlm model -valid valid -hidden 50\n");
    	fprintf(stdout,"rnnlm -rnnlm model -test test\n");
        fprintf(stdout,"rnnlm -rnnlm model -compress 4 -test test\n");
    	fprintf(stdout,"\n");

    	return 0;	//***
    }


    //set debug mode
    i=argPos((char *)"-debug", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: debug mode not specified!\n");
            return 0;
        }

        debug_mode=atoi(argv[i+1]);

	if (debug_mode>0)
        fprintf(stderr,"debug mode: %d\n", debug_mode);
    }


    //search for train file
    i=argPos((char *)"-train", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: training data file not specified!\n");
            return 0;
        }

        strcpy(train_file, argv[i+1]);

	if (debug_mode>0)
        fprintf(stderr,"train file: %s\n", train_file);

        f=fopen(train_file, "rb");
        if (f==NULL) {
            fprintf(stderr,"ERROR: training data file not found!\n");
            return 0;
        }

        train_mode=1;

        train_file_set=1;
    }


    //set one-iter
    i=argPos((char *)"-one-iter", argc, argv);
    if (i>0) {
        one_iter=1;

        if (debug_mode>0)
        fprintf(stderr,"Training for one iteration\n");
    }


    //search for validation file
    i=argPos((char *)"-valid", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: validation data file not specified!\n");
            return 0;
        }

        strcpy(valid_file, argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"valid file: %s\n", valid_file);

        f=fopen(valid_file, "rb");
        if (f==NULL) {
            fprintf(stderr,"ERROR: validation data file not found!\n");
            return 0;
        }

        valid_data_set=1;
    }

    if (train_mode && !valid_data_set) {
	if (one_iter==0) {
	    fprintf(stderr,"ERROR: validation data file must be specified for training!\n");
    	    return 0;
    	}
    }


    //set nbest rescoring mode
    i=argPos((char *)"-nbest", argc, argv);
    if (i>0) {
	nbest=1;
        if (debug_mode>0)
        fprintf(stderr,"Processing test data as list of nbests\n");
    }


    //search for test file
    i=argPos((char *)"-test", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: test data file not specified!\n");
            return 0;
        }

        strcpy(test_file, argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"test file: %s\n", test_file);


	if (nbest && (!strcmp(test_file, "-"))) ; else {
            f=fopen(test_file, "rb");
            if (f==NULL) {
                fprintf(stderr,"ERROR: test data file not found!\n");
                return 0;
            }
        }

        test_data_set=1;
    }


    //set class size parameter
    i=argPos((char *)"-class", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: amount of classes not specified!\n");
            return 0;
        }

        class_size=atoi(argv[i+1]);

	if (debug_mode>0)
        fprintf(stderr,"class size: %d\n", class_size);
    }


    //set old class
    i=argPos((char *)"-old-classes", argc, argv);
    if (i>0) {
        old_classes=1;

	if (debug_mode>0)
        fprintf(stderr,"Old algorithm for computing classes will be used\n");
    }


    //set lambda
    i=argPos((char *)"-lambda", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: lambda not specified!\n");
            return 0;
        }

        lambda=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Lambda (interpolation coefficient between rnnlm and other lm): %f\n", lambda);
    }


    //set gradient cutoff
    i=argPos((char *)"-gradient-cutoff", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: gradient cutoff not specified!\n");
            return 0;
        }

        gradient_cutoff=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Gradient cutoff: %f\n", gradient_cutoff);
    }


    //set dynamic
    i=argPos((char *)"-dynamic", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: dynamic learning rate not specified!\n");
            return 0;
        }

        dynamic=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Dynamic learning rate: %f\n", dynamic);
    }


    //set compress
    i=argPos((char *)"-compress", argc, argv);
    if (i>0) {
        if (i+1==argc) {
          fprintf(stderr, "ERROR: number of bits not specified!\n");
          return 0;
        }

        ncluster=atoi(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Number of clustering bits: %d\n", ncluster);
    }

    //set kmean
    i=argPos((char *)"-kmean", argc, argv);
    if (i>0) {
      if (i+1==argc) {
        fprintf(stderr,"ERROR: number of iterations not specified!\n");
        return 0;
      }
      
      kmean_iter=atoi(argv[i+1]);

      if (debug_mode>0)
        fprintf(stderr,"Using kmean quantization with given number of iterations: %d\n", kmean_iter);
    }

    //set write-compressed
    i=argPos((char *)"-write-compressed", argc, argv);
    if (i>0) {
      if (i+1==argc) {
        fprintf(stderr,"ERROR: compressed model filename not specified!\n");
        return 0;
      }

      strcpy(compress_file,argv[i+1]);

      if (debug_mode>0)
        fprintf(stderr,"Writing compressed model to: %s\n", compress_file);
    }

    //set gen
    i=argPos((char *)"-gen", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: gen parameter not specified!\n");
            return 0;
        }

        gen=atoi(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Generating # words: %d\n", gen);
    }


    //set independent
    i=argPos((char *)"-independent", argc, argv);
    if (i>0) {
        independent=1;

        if (debug_mode>0)
        fprintf(stderr,"Sentences will be processed independently...\n");
    }


    //set learning rate
    i=argPos((char *)"-alpha", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: alpha not specified!\n");
            return 0;
        }

        starting_alpha=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Starting learning rate: %f\n", starting_alpha);
        alpha_set=1;
    }


    //set regularization
    i=argPos((char *)"-beta", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: beta not specified!\n");
            return 0;
        }

        regularization=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Regularization: %f\n", regularization);
    }


    //set min improvement
    i=argPos((char *)"-min-improvement", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: minimal improvement value not specified!\n");
            return 0;
        }

        min_improvement=atof(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Min improvement: %f\n", min_improvement);
    }


    //set anti kasparek
    i=argPos((char *)"-anti-kasparek", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: anti-kasparek parameter not set!\n");
            return 0;
        }

        anti_k=atoi(argv[i+1]);

        if ((anti_k!=0) && (anti_k<10000)) anti_k=10000;

        if (debug_mode>0)
        fprintf(stderr,"Model will be saved after each # words: %d\n", anti_k);
    }


    //set hidden layer size
    i=argPos((char *)"-hidden", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: hidden layer size not specified!\n");
            return 0;
        }

        hidden_size=atoi(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Hidden layer size: %d\n", hidden_size);
    }


    //set compression layer size
    i=argPos((char *)"-compression", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: compression layer size not specified!\n");
            return 0;
        }

        compression_size=atoi(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Compression layer size: %d\n", compression_size);
    }


    //set direct connections
    i=argPos((char *)"-direct", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: direct connections not specified!\n");
            return 0;
        }

        direct=atoi(argv[i+1]);

        direct*=1000000;
	if (direct<0) direct=0;

        if (debug_mode>0)
        fprintf(stderr,"Direct connections: %dM\n", (int)(direct/1000000));
    }


    //set order of direct connections
    i=argPos((char *)"-direct-order", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: direct order not specified!\n");
            return 0;
        }

        direct_order=atoi(argv[i+1]);
        if (direct_order>MAX_NGRAM_ORDER) direct_order=MAX_NGRAM_ORDER;

        if (debug_mode>0)
        fprintf(stderr,"Order of direct connections: %d\n", direct_order);
    }


    //set bptt
    i=argPos((char *)"-bptt", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: bptt value not specified!\n");
            return 0;
        }

        bptt=atoi(argv[i+1]);
        bptt++;
        if (bptt<1) bptt=1;

        if (debug_mode>0)
        fprintf(stderr,"BPTT: %d\n", bptt-1);
    }


    //set bptt block
    i=argPos((char *)"-bptt-block", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: bptt block value not specified!\n");
            return 0;
        }

        bptt_block=atoi(argv[i+1]);
        if (bptt_block<1) bptt_block=1;

        if (debug_mode>0)
        fprintf(stderr,"BPTT block: %d\n", bptt_block);
    }


    //set random seed
    i=argPos((char *)"-rand-seed", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: Random seed variable not specified!\n");
            return 0;
        }

        rand_seed=atoi(argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"Rand seed: %d\n", rand_seed);
    }


    //use other lm
    i=argPos((char *)"-lm-prob", argc, argv);
    if (i>0) {
	if (i+1==argc) {
            fprintf(stderr,"ERROR: other lm file not specified!\n");
            return 0;
        }

        strcpy(lmprob_file, argv[i+1]);

        if (debug_mode>0)
        fprintf(stderr,"other lm probabilities specified in: %s\n", lmprob_file);

        f=fopen(lmprob_file, "rb");
        if (f==NULL) {
            fprintf(stderr,"ERROR: other lm file not found!\n");
            return 0;
        }

        use_lmprob=1;
    }


    //search for binary option
    i=argPos((char *)"-binary", argc, argv);
    if (i>0) {
        if (debug_mode>0)
        fprintf(stderr,"Model will be saved in binary format\n");

        fileformat=BINARY;
    }


    //search for rnnlm file
    i=argPos((char *)"-rnnlm", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            fprintf(stderr,"ERROR: model file not specified!\n");
            return 0;
        }

        strcpy(rnnlm_file, argv[i+1]);

        if (debug_mode>0)
       fprintf(stderr,"rnnlm file: %s\n", rnnlm_file);

        f=fopen(rnnlm_file, "rb");

        rnnlm_file_set=1;
    }
    if (train_mode && !rnnlm_file_set) {
    	fprintf(stderr,"ERROR: rnnlm file must be specified for training!\n");
    	return 0;
    }
    if (test_data_set && !rnnlm_file_set) {
    	fprintf(stderr,"ERROR: rnnlm file must be specified for testing!\n");
    	return 0;
    }
    if (!test_data_set && !train_mode && gen==0) {
    	fprintf(stderr,"ERROR: training or testing must be specified!\n");
    	return 0;
    }
    if ((gen>0) && !rnnlm_file_set) {
	fprintf(stderr,"ERROR: rnnlm file must be specified to generate words!\n");
    	return 0;
    }


    srand(1);

    if (train_mode) {
    	CRnnLM model1;

    	model1.setTrainFile(train_file);
    	model1.setRnnLMFile(rnnlm_file);
    	model1.setFileType(fileformat);

    	model1.setOneIter(one_iter);
    	if (one_iter==0) model1.setValidFile(valid_file);

	model1.setClassSize(class_size);
	model1.setOldClasses(old_classes);
    	model1.setLearningRate(starting_alpha);
    	model1.setGradientCutoff(gradient_cutoff);
    	model1.setRegularization(regularization);
    	model1.setMinImprovement(min_improvement);
    	model1.setHiddenLayerSize(hidden_size);
    	model1.setCompressionLayerSize(compression_size);
    	model1.setDirectSize(direct);
    	model1.setDirectOrder(direct_order);
    	model1.setBPTT(bptt);
    	model1.setBPTTBlock(bptt_block);
    	model1.setRandSeed(rand_seed);
    	model1.setDebugMode(debug_mode);
    	model1.setAntiKasparek(anti_k);
	model1.setIndependent(independent);

    	model1.alpha_set=alpha_set;
    	model1.train_file_set=train_file_set;

    	model1.trainNet();
    }

    if (test_data_set && rnnlm_file_set) {
        CRnnLM model1;

        model1.setLambda(lambda);
        model1.setRegularization(regularization);
        model1.setDynamic(dynamic);
        model1.setTestFile(test_file);
        model1.setRnnLMFile(rnnlm_file);
        model1.setRandSeed(rand_seed);
        model1.useLMProb(use_lmprob);
        if (use_lmprob) model1.setLMProbFile(lmprob_file);
        model1.setDebugMode(debug_mode);

	if (nbest==0) {
          // FIXME this is an ugly hack!
          if (ncluster!=0) {
            model1.setNCluster(ncluster);
            if (kmean_iter>0) model1.setKMean(kmean_iter);
            model1.setFileType(COMPRESSED);
          }
          model1.testNet();
          if (compress_file[0]!=0) {
            model1.setRnnLMFile(compress_file);
            model1.saveNet();
          }
        }
	else model1.testNbest();
    }

    if (gen>0) {
	CRnnLM model1;

	model1.setRnnLMFile(rnnlm_file);
	model1.setDebugMode(debug_mode);
	model1.setRandSeed(rand_seed);
	model1.setGen(gen);

	model1.testGen();
    }


    return 0;
}
