#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>

typedef struct options_t
{
	// true is classify, false is regression
	bool classification;
	bool regression;
	bool has_header;
	size_t label_column;
	size_t num_columns;
	size_t k_nearest_neighbors;
	bool label_defined;
	char* filename;
	size_t num_threads;
} options_t;

typedef struct data_point_t
{
	size_t num_features;
	long double* features;
	long double output_feature;
	char* label;
	bool label_is_double;
} data_point_t;

typedef struct training_data_t
{
	size_t num_samples;
	data_point_t* samples;
} training_data_t;

typedef struct distance_t
{
	double distance;
	data_point_t* paired_data;
} distance_t;

int comp_dist( const void* a, const void* b )
{
	double d_1 = ( (distance_t*)a )->distance;
	double d_2 = ( (distance_t*)b )->distance;
	if ( d_1 < d_2 )
	{
		return -1;
	}
	else if ( d_1 > d_2 )
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

void usage( char* prog_name, int exit_code )
{
	fprintf(
	  stderr,
	  // It's an ugly string but clang-format destroyed it *shrug*
	  "USAGE:"
	  "    %s [FLAGS and OPTIONS] FILE\n"
	  "\n"
	  "Supplied data should be given in the same order/format as the input file, "
	  "eg a\n"
	  "csv file with 2 real values, the label, then 2 more real values, a "
	  "single data\n"
	  "point should be like so:\n"
	  "\n"
	  "    real1,real2,real3,real4\n"
	  "\n"
	  "ARGUMENTS:"
	  "\n"
	  "    FILE    The name of a comma or tab separated value file, in which the "
	  "first\n"
	  "            row can be the labels, which will be ignored. The specified "
	  "file\n"
	  "            should not have more than one non-real field/column, which "
	  "should\n"
	  "            be the label for that data entry. If any columns have a "
	  "label, you\n"
	  "            must use one of the classification flags (-c or "
	  "--classification).\n"
	  "            Otherwise, use the regression flags (-r or --regression). If "
	  "you\n"
	  "            are classifying a file that does not have any labels, you "
	  "must use\n"
	  "            the label option (-l or --label) to specify a column number "
	  "to use\n"
	  "            as the label (0-indexed)."
	  "\n"
	  "\n"
	  "FLAGS:\n"
	  "\n"
	  "    -c, --classification     Classify data read from stdin\n"
	  "\n"
	  "    -r, --regression         Use a regression of the data in FILE to "
	  "predict the\n"
	  "                             value of the dependent variable in the "
	  "specified\n"
	  "                             column (-l or --label is required)"
	  "\n"
	  "    -h, --help               Show this help text\n"
	  "\n"
	  "OPTIONS:\n"
	  "\n"
	  "    -l, --label-column       Column number to use as the label for "
	  "regression;\n"
	  "                             required when using the -r/--regression "
	  "flag\n"
	  "\n"
	  "    -k, --k-nearest          Number of nearest neighbors to use when\n"
	  "                             classifying or performing regression on an "
	  "input\n"
	  "                             data point -- default is 5\n"
	  "    -t, --threads            Number of threads to use -- defaults to 4\n",
	  prog_name );
	exit( exit_code );
}

options_t process_args( int argc, char* argv[] )
{
	for ( int i = 0; i < argc; ++i )
	{
		if ( strcmp( argv[i], "-h" ) == 0 || strcmp( argv[i], "--help" ) == 0 )
			usage( argv[0], EX_OK );
	}

	// Setup defaults
	options_t opts;
	opts.classification = false;
	opts.regression = false;
	opts.label_column = -1;
	opts.label_defined = false;
	opts.k_nearest_neighbors = 5;
	opts.num_threads = 4;

	bool reg_flag = false, class_flag = false, label_flag = false;

	for ( int i = 1; i < argc - 1; ++i )
	{
		if ( strcmp( "-c", argv[i] ) == 0 ||
		     strcmp( "--classification", argv[i] ) == 0 )
		{
			opts.classification = true;
			opts.regression = false;
			class_flag = true;
		}
		else if ( strcmp( "-r", argv[i] ) == 0 ||
		          strcmp( "--regression", argv[i] ) == 0 )
		{
			opts.regression = true;
			opts.classification = false;
			reg_flag = true;
		}
		else if ( ( strcmp( "-l", argv[i] ) == 0 ||
		            strcmp( "--label-column", argv[i] ) == 0 ) &&
		          i + 1 < argc )
		{
			opts.label_column = atoi( argv[i + 1] );
			opts.label_defined = true;
			label_flag = true;
		}
		else if ( ( strcmp( "-k", argv[i] ) == 0 ||
		            strcmp( "--k-nearest", argv[i] ) == 0 ) &&
		          i + 1 < argc )
		{
			opts.k_nearest_neighbors = atoi( argv[i + 1] );
		}
		else if ( ( strcmp( "-t", argv[i] ) == 0 ||
		            strcmp( "--threads", argv[i] ) == 0 ) &&
		          i + 1 < argc )
		{
			opts.num_threads = atoi( argv[i + 1] );
		}
	}

	if ( class_flag && reg_flag )
	{
		fprintf( stderr,
		         "Both classification and regression were passed as flags\n"
		         "Use one or the other\n" );
		exit( EX_USAGE );
	}
	else if ( reg_flag && !label_flag )
	{
		fprintf( stderr,
		         "Regression and label column flags are required together\n" );
		exit( EX_USAGE );
	}
	else if ( class_flag && label_flag )
	{
		fprintf(
		  stderr,
		  "A column should not be defined if using classification\n"
		  "If the classifier is an number, change it to a string (man sed)" );
		exit( EX_USAGE );
	}

	// If we have gotten this far, it's probably a good set of args and we can get
	// the filename
	opts.filename = argv[argc - 1];

	return opts;
}

bool is_label( char* split )
{
	bool is_label = false;

	for ( int i = 0; i < strlen( split ) && !is_label; ++i )
	{
		if ( !isdigit( split[i] ) && split[i] != '.' && split[i] != '-' )
		{
			// Shortcut out
			return true;
		}
	}

	return false;
}

uint64_t count_lines( FILE* file, options_t* options )
{
	uint16_t lines = 0;
	while ( EOF != ( fscanf( file, "%*[^\n]" ), fscanf( file, "%*c" ) ) ) ++lines;

	fseek( file, 0, SEEK_SET );

	char* line = NULL;
	size_t max_line_len = 0;

	getline( &line, &max_line_len, file );

	char* split = strtok( line, "," );
	size_t non_real_in_first_line = 0;
	size_t columns = 0;

	while ( split != NULL )
	{
		if ( is_label( split ) )
		{
			non_real_in_first_line++;
		}
		columns++;
		split = strtok( NULL, "," );
	}

	if ( non_real_in_first_line >= 2 )
	{
		// We can ignore the first line when we allocate space
		lines--;
		options->has_header = true;
	}
	else
	{
		options->has_header = false;
	}
	options->num_columns = columns;

	// lets detect the label column too
	getline( &line, &max_line_len, file );
	line[strlen( line ) - 1] = 0;  // Fix dat newline
	split = strtok( line, "," );
	size_t non_real_data_columns = 0;
	size_t column_num = 0;

	while ( split != NULL )
	{
		if ( is_label( split ) )
		{
			non_real_data_columns++;
			if ( options->label_defined && options->label_column != column_num )
			{
				fprintf( stderr,
				         "Given label column does not match label in data"
				         " (%zu != %zu, Data = %s)\n",
				         column_num, options->label_column, split );
				exit( EX_DATAERR );
			}
			options->label_column = column_num;
			options->label_defined = true;
			options->classification = true;
		}
		column_num++;
		split = strtok( NULL, "," );
	}

	if ( options->regression && non_real_data_columns > 0 )
	{
		fprintf( stderr, "Regression specified but data file has non-real data\n" );
		exit( EX_DATAERR );
	}
	else if ( non_real_data_columns > 1 )
	{
		fprintf( stderr, "Data file has more than one non-real data column\n" );
		exit( EX_DATAERR );
	}

	fseek( file, 0, SEEK_SET );

	return lines;
}

data_point_t parse_training_line( char* line, options_t* opts )
{
	// Get rid of the darn trailing newline
	if ( line[strlen( line ) - 1] == '\n' )
	{
		line[strlen( line ) - 1] = '\0';
	}

	char* split = strtok( line, "," );
	size_t column_num = 0;
	size_t feature_num = 0;
	size_t num_labels = 0;

	data_point_t data_point;
	data_point.num_features = opts->num_columns - 1;
	data_point.features = malloc( sizeof( long double ) * opts->num_columns - 1 );

	while ( split != NULL )
	{
		// This column is the label column
		if ( opts->label_column == column_num )
		{
			if ( opts->classification )
			{
				data_point.label = strdup( split );

				data_point.output_feature = FP_NAN;
				data_point.label_is_double = false;
			}
			else
			{
				data_point.output_feature = strtod( split, NULL );
				data_point.label = split;
				data_point.label_is_double = true;
				feature_num++;
			}
		}
		else  // This column is a feature
		{
			data_point.features[feature_num] = strtod( split, NULL );
			feature_num++;
		}
		column_num++;

		split = strtok( NULL, "," );
	}

	// This should never happen if we find a single valid label or
	// the label column was defined in the args
	if ( !opts->label_defined )
	{
		if ( num_labels == 0 && opts->classification )
		{
			fprintf(
			  stderr,
			  "No reasonable label was found in the data for a classification" );
			exit( EX_DATAERR );
		}
		else
		{
			fprintf( stderr, "No label column defined for regression\n" );
			exit( EX_DATAERR );
		}
	}

	return data_point;
}

data_point_t parse_query_line( char* line, size_t num_features )
{
	data_point_t data_pt;
	data_pt.num_features = num_features;
	data_pt.features = malloc( sizeof( long double ) * num_features );
	data_pt.label_is_double = true;
	data_pt.label = NULL;
	data_pt.output_feature = 0;

	char* split = strtok( line, "," );

	size_t feature_num = 0;
	while ( split != NULL )
	{
		data_pt.features[feature_num] = strtod( split, NULL );

		split = strtok( NULL, "," );
		feature_num++;
	}

	return data_pt;
}

void parse_file( training_data_t* data, FILE* file, options_t opts )
{
	char* curr_line;
	size_t last_len = 0;
	ssize_t line_len;
	if ( opts.has_header )
	{
		// Skip the first line
		line_len = getline( &curr_line, &last_len, file );
	}

	size_t sample_num = 0;
	while ( ( line_len = getline( &curr_line, &last_len, file ) ) != -1 )
	{
		data->samples[sample_num] = parse_training_line( curr_line, &opts );
		sample_num++;
	}
}

void print_data_point( data_point_t data_pt )
{
	if ( data_pt.label_is_double )
	{
		printf( "Label (double): %Lf\nFeatures: ", data_pt.output_feature );
	}
	else
	{
		printf( "Label (string): %s\nFeatures: ", data_pt.label );
	}

	for ( int i = 0; i < data_pt.num_features; ++i )
	{
		printf( "%Lf ", data_pt.features[i] );
	}
	printf( "\n" );
}

void free_training_data( training_data_t data )
{
	for ( int i = 0; i < data.num_samples; ++i )
	{
		free( data.samples[i].features );
		if ( !data.samples[i].label_is_double )
		{
			free( data.samples[i].label );
		}
	}

	free( data.samples );
}

distance_t euclid_dist( data_point_t* training_pt, data_point_t* query_pt )
{
	distance_t distance;
	distance.distance = 0;
	if ( training_pt->num_features != query_pt->num_features )
	{
		fprintf( stderr, "Read data point has improper number of features\n" );
		exit( EX_DATAERR );
	}

	for ( int i = 0; i < training_pt->num_features; ++i )
	{
		distance.distance +=
		  pow( training_pt->features[i] - query_pt->features[i], 2 );
	}

	distance.distance = pow( distance.distance, 0.5 );
	distance.paired_data = training_pt;

	return distance;
}

char* find_classification( distance_t* distances, size_t num_neighbors )
{
	int* counts = calloc( num_neighbors, sizeof( int ) );
	size_t num_unique_labels = 0;
	char** unique_labels = calloc( num_neighbors, sizeof( char* ) );

	for ( int i = 0; i < num_neighbors; ++i )
	{
		char* neighbor_label = distances[i].paired_data->label;

		// For the first iteration where we have to put in at least one label
		if ( num_unique_labels == 0 )
		{
			unique_labels[0] = strdup( neighbor_label );
			counts[0]++;
			num_unique_labels++;
		}
		else
		{
			bool found = false;
			for ( int j = 0; !found && j < num_unique_labels; ++j )
			{
				if ( strcmp( neighbor_label, unique_labels[j] ) == 0 )
				{
					counts[j]++;

					// break out
					found = true;
				}
			}

			if ( !found )
			{
				unique_labels[i] = strdup( neighbor_label );
				num_unique_labels++;
				counts[i]++;
			}
		}
	}

	int max = counts[0];
	size_t max_index = 0;
	for ( int i = 1; i < num_unique_labels; ++i )
	{
		if ( counts[i] > max )
		{
			max = counts[i];
			max_index = i;
		}
	}

	char* return_label = strdup( unique_labels[max_index] );

	for ( int i = 0; i < num_unique_labels; ++i )
	{
		free( unique_labels[i] );
	}
	free( unique_labels );
	free( counts );

	return return_label;
}

void knn( training_data_t* data, options_t opts )
{
	printf(
	  "Training data parsed\nReading input queries in same format as input "
	  "file, one query per line\n"
	  "Use Ctrl-D to end queries\n" );

	char* line;
	size_t last_len = 0;
	ssize_t line_len;

	while ( ( line_len = getline( &line, &last_len, stdin ) ) != -1 )
	{
		data_point_t query_data = parse_query_line( line, opts.num_columns - 1 );

		distance_t* distances;
		distances = malloc( sizeof( distance_t ) * data->num_samples );

		// We use a big block when scheduling so that each CPU hopefully does all the
		// tasks one after another and keeps nearby data in cache, but so much of it
		// is in the heap behind pointers that it probably makes little difference
#pragma omp parallel for schedule( static,                                    \
                                   2 * data->num_samples / opts.num_threads ) \
  num_threads( opts.num_threads )
		for ( int i = 0; i < data->num_samples; ++i )
		{
			distances[i] = euclid_dist( &data->samples[i], &query_data );
		}

		qsort( distances, data->num_samples, sizeof( *distances ), comp_dist );

		if ( opts.classification )
		{
			char* classification =
			  find_classification( distances, opts.k_nearest_neighbors );
			printf( "Predicted data point classification: %s\n", classification );
			free( classification );
		}
		else
		{
			long double regression_avg = 0;
			for ( int i = 0; i < opts.k_nearest_neighbors; ++i )
			{
				regression_avg += distances[i].paired_data->output_feature;
			}
			regression_avg /= opts.k_nearest_neighbors;
			printf( "Predicted output feature: %Lf\n", regression_avg );
		}

		free( distances );
	}
}

int main( int argc, char* argv[] )
{
	if ( argc < 2 )
	{
		usage( argv[0], EX_USAGE );
	}

	options_t options = process_args( argc, argv );

	FILE* data_file = fopen( options.filename, "r" );
	if ( data_file == NULL )
	{
		fprintf( stderr, "File '%s' does not exist or could not be found\n",
		         options.filename );
		exit( EX_IOERR );
	}

	training_data_t training_data;
	training_data.num_samples = count_lines( data_file, &options );

	training_data.samples =
	  malloc( sizeof( data_point_t ) * training_data.num_samples );

	parse_file( &training_data, data_file, options );

#ifdef DEBUG
	for ( int i = 0; i < training_data.num_samples; ++i )
	{
		print_data_point( training_data.samples[i] );
	}
#endif

	fclose( data_file );

	knn( &training_data, options );

	free_training_data( training_data );

	return 0;
}
