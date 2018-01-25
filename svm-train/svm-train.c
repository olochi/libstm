#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
//#include <string>
#include "..\libstm\svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"	5 -- SVDD		(C should be between 1/num_instances and 1)\n"
	"	6 -- R^2: L1SVM\n"
	"	7 -- R^2: L2SVM\n"
	/****************************************************************
	* 作    者：ake
	* 描述说明：8--STDD,文件名格式，Yale_64_64;
	*****************************************************************/
	"	8 -- STDD		(C should be between 1/num_instances and 1)\n"
	/****************************************************************
	* end
	*****************************************************************/
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of -s 0, 3, 4, 5 and 7 (default 1, except 2/num_instances for -s 5)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	////////////////////////////k
	char ch;
	printf("Please enter a character.\n");
	scanf("%c", &ch);   /* user inputs character */

	/////////////////////////////
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int agrc1, char **agrv1, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

//int main(int agrc, char **agrv)
int main(int agrc, char **agrv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	
	
	//////////////
	//input_file_name = {'s','v','m','g','u','i','d','e','1'};
	//string tem = "sdf";
	//input_file_name[0] = 's';
	//input_file_name[1] = 'g';
	//input_file_name[2] = '1';
	//input_file_name[3] = '\0';
	//int agrc1 = 0;
	int agrc_tem = 0;
	//char *agrv1[2];
	//model = svmtrain(labels, data, '-s 5 -t 2 -c 0.5 -g 0.001 -nu 0.1');
	//参数个数 = 参数*2 + 文件名
	agrc_tem = agrc + 8 +1;
	char *agrv_tem[12];

	
	//////////////参数1
	char agrv_1[10];
	agrv_1[0] = '-';
	agrv_1[1] = 's';
	agrv_1[2] = '\0';

	int agr_1 = 5;
	char agrv_1_[10];
	_itoa_s(agr_1, agrv_1_, sizeof(agrv_1_), 10);

	agrv_tem[1] = agrv_1;
	agrv_tem[2] = agrv_1_;
	//////////////参数2	
	char agrv_2[10];
	agrv_2[0] = '-';
	agrv_2[1] = 't';
	agrv_2[2] = '\0';

	int agr_2 = 2;
	char agrv_2_[10];
	_itoa_s(agr_2, agrv_2_, sizeof(agrv_2_), 10);

	agrv_tem[3] = agrv_2;
	agrv_tem[4] = agrv_2_;
	//////////////参数3
	char agrv_3[10];
	agrv_3[0] = '-';
	agrv_3[1] = 'c';
	agrv_3[2] = '\0';

	double agr_3 = 0.3126;
	char agrv_3_[10];
	//_gcvt_s(agrv_3_, sizeof(agrv_3_), agr_3, 3);
	sprintf(agrv_3_, "%lf", agr_3);

	agrv_tem[5] = agrv_3;
	agrv_tem[6] = agrv_3_;
	//////////////参数4
	char agrv_4[10];
	agrv_4[0] = '-';
	agrv_4[1] = 'g';
	agrv_4[2] = '\0';

	double agr_4 = 0.0078125;
	char agrv_4_[10];
	//agrv_4_[31] = '\0';
	//_gcvt_s(agrv_4_, sizeof(agrv_4_), agr_4, 3);
	sprintf(agrv_4_, "%lf", agr_4);
	double abc = atof(agrv_4_);
	agrv_tem[7] = agrv_4;
	agrv_tem[8] = agrv_4_;
	//////////////参数5
	/*char agrv_5[10];
	agrv_5[0] = '-';
	agrv_5[1] = 'n';
	agrv_5[2] = 'u';
	agrv_5[3] = '\0';

	double agr_5 = 0.1;
	char agrv_5_[10];
	_gcvt_s(agrv_5_, sizeof(agrv_5_), agr_5, 3);
	

	agrv_tem[9] = agrv_5;
	agrv_tem[10] = agrv_5_;*/
	/////////////训练文件名
	
	char agrv_file_train[10];
	agrv_file_train[0] = 's';
	agrv_file_train[1] = 'd';
	//agrv_file_train[2] = '1';
	agrv_file_train[2] = '\0';
	//////////////////////////
	
	
	agrv_tem[0] = *agrv;
	agrv_tem[9] = agrv_file_train;
	///////////////////
	
	///////////////////
	parse_command_line(agrc_tem, agrv_tem, input_file_name, model_file_name);
	//parse_command_line(agrc, agrv, input_file_name, model_file_name);


	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		/////////////////ake
		char ch;
		printf("Please enter a character to exit.\n");
		scanf("%c", &ch);   /* user inputs character */
		exit(1);
	}


	if(cross_validation)
	{
		if(param.svm_type == R2 || param.svm_type == R2q)
			fprintf(stderr, "\"R^2\" cannot do cross validation.\n");
		else
			do_cross_validation();
	}
	else
	{
		model = svm_train(&prob,&param);
		if(param.svm_type == R2 || param.svm_type == R2q)
			fprintf(stderr, "\"R^2\" does not generate a model.\n");
		else if(svm_save_model(model_file_name,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			/////////////////ake
			char ch;
			printf("Please enter a character to exit.\n");
			scanf("%c", &ch);   /* user inputs character */
			exit(1);
		}
		svm_free_and_destroy_model(&model);
	}
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	/////////////////ake
	char ch;
	printf("Please enter a character to exit.\n");
	scanf("%c", &ch);   /* user inputs character */

	/////////////////
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

void parse_command_line(int agrc1, char **agrv1, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 0;	// 1 or 2/prob.l
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	// parse options
	for(i=1;i<agrc1;i++)
	{
		if(agrv1[i][0] != '-') break;
		if(++i>=agrc1)
			exit_with_help();
		switch(agrv1[i-1][1])
		{
			case 's':
				param.svm_type = atoi(agrv1[i]);
				break;
			case 't':
				param.kernel_type = atoi(agrv1[i]);
				break;
			case 'd':
				param.degree = atoi(agrv1[i]);
				break;
			case 'g':
				param.gamma = atof(agrv1[i]);
				break;
			case 'r':
				param.coef0 = atof(agrv1[i]);
				break;
			case 'n':
				param.nu = atof(agrv1[i]);
				break;
			case 'm':
				param.cache_size = atof(agrv1[i]);
				break;
			case 'c':
				param.C = atof(agrv1[i]);
				break;
			case 'e':
				param.eps = atof(agrv1[i]);
				break;
			case 'p':
				param.p = atof(agrv1[i]);
				break;
			case 'h':
				param.shrinking = atoi(agrv1[i]);
				break;
			case 'b':
				param.probability = atoi(agrv1[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(agrv1[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&agrv1[i-1][2]);
				param.weight[param.nr_weight-1] = atof(agrv1[i]);
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", agrv1[i-1][1]);
				exit_with_help();
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=agrc1)
		exit_with_help();

	strcpy(input_file_name, agrv1[i]);

	if(i<agrc1-1)
		strcpy(model_file_name,agrv1[i+1]);
	else
	{
		char *p = strrchr(agrv1[i],'/');
		if(p==NULL)
			p = agrv1[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
}

// read in a problem (in svmlight format)

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;
	if(param.C == 0)
	{
		if (param.svm_type == SVDD && prob.l > 0)
			param.C = 2.0/prob.l;
		else
			param.C = 1;
	}

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
