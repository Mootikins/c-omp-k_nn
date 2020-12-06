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
	long double distance;
	data_point_t* other_data_point;
} distance_t;

typedef struct knn_distances_t
{
	size_t num_distances;
	distance_t* distances;
} knn_distances_t;

int comp_dist( const void* dist1, const void* dist2 )
{
	distance_t d_1 = *( (distance_t*)dist1 );
	distance_t d_2 = *( (distance_t*)dist2 );
	if ( d_1.distance > d_2.distance ) return 1;
	if ( d_1.distance < d_2.distance ) return -1;
	return 0;
}

void usage( char* prog_name, int exit_code )
{
	fprintf(
	  stderr,
	  // It's an ugly string but clang-format destroyed it *shrug*
	  "USAGE:"
	  "    %s [FLAGS and OPTIONS] FILE"
	  "\n"
	  "Supplied data should be given in the same order/format as the input file, "
	  "eg a"
	  "csv file  with 2 real values, the label, then 2 more real values, a "
	  "single data"
	  "point should be like so:"
	  "\n"
	  "    real1,real2,real3,real4"
	  "\n"
	  "ARGUMENTS:"
	  "\n"
	  "    FILE    The name of a comma or tab separated value file, in which the "
	  "first"
	  "            row can be the labels, which will be ignored. The specified "
	  "file"
	  "            should not have more than one non-real field/column, which "
	  "should"
	  "            be the label for that data entry. If any columns have a "
	  "label, you"
	  "            must use one of the classification flags (-c or "
	  "--classification)."
	  "            Otherwise, use the regression flags (-r or --regression). If "
	  "you"
	  "            are classifying a file that does not have any labels, you "
	  "must use"
	  "            the label option (-l or --label) to specify a column number "
	  "to use"
	  "            as the label (0-indexed)."
	  "\n"
	  "\n"
	  "FLAGS:"
	  "\n"
	  "    -c, --classification     Classify data read from stdin"
	  "\n"
	  "    -r, --regression         Use a regression of the data in FILE to "
	  "predict the"
	  "                             value of the dependent variable in the "
	  "specified"
	  "                             column (-l or --label is required)"
	  "\n"
	  "    -h, --help               Show this help text"
	  "\n"
	  "OPTIONS:"
	  "\n"
	  "    -l, --label-column       Column number to use as the label for "
	  "regression;"
	  "                             required when using the -r/--regression flag"
	  "\n"
	  "    -k, --k-nearest          Number of nearest neighbors to use when"
	  "                             classifying or performing regression on an "
	  "input"
	  "                             data point -- default is 5",
	  prog_name );
	exit( exit_code );
}

options_t process_args( int argc, char* argv[] )
{
	for ( int i = 0; i < argc; ++i )
	{
		if ( strcmp( argv[i], "-h" ) == 0 || strcmp( argv[i], "--help" ) == 0 )
			exit( EX_OK );
	}

	// Setup defaults
	options_t opts;
	opts.classification = false;
	opts.regression = false;
	opts.label_column = -1;
	opts.label_defined = false;
	opts.k_nearest_neighbors = 5;

	for ( int i = 1; i < argc - 1; ++i )
	{
		if ( strcmp( "-c", argv[i] ) == 0 ||
		     strcmp( "--classification", argv[i] ) == 0 )
		{
			opts.classification = true;
		}
		else if ( strcmp( "-r", argv[i] ) == 0 ||
		          strcmp( "--regression", argv[i] ) == 0 )
		{
			opts.regression = true;
		}
		else if ( ( strcmp( "-l", argv[i] ) == 0 ||
		            strcmp( "--label-column", argv[i] ) == 0 ) &&
		          i + 1 < argc )
		{
			printf( "Read label column: %s", argv[i] );
			opts.label_column = atoi( argv[i + 1] );
			opts.label_defined = true;
		}
		else if ( ( strcmp( "-k", argv[i] ) == 0 ||
		            strcmp( "--k-nearest", argv[i] ) == 0 ) &&
		          i + 1 < argc )
		{
			opts.k_nearest_neighbors = atoi( argv[i + 1] );
		}
	}

	if ( opts.classification && opts.regression )
	{
		fprintf( stderr,
		         "Both classification and regression were passed as flags -- use "
		         "one or the other\n\n" );
		exit( EX_USAGE );
	}

	if ( opts.regression && !opts.label_defined )
	{
		fprintf( stderr,
		         "Regression and label column flags are required together\n" );
		exit( EX_USAGE );
	}

	// If we have gotten this far, it's probably a good set of args and we can get
	// the filename
	opts.filename = argv[argc - 1];

	return opts;
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
	size_t not_real_number_columns = 0;
	size_t columns = 0;

	while ( split != NULL )
	{
		if ( isalpha( split[0] ) || split[0] == '.' )
		{
			not_real_number_columns++;
		}
		columns++;
		split = strtok( NULL, "," );
	}

	if ( not_real_number_columns >= 2 )
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
		bool is_label = false;

		for ( int i = 0; i < strlen( split ) && !is_label; ++i )
		{
			if ( !isdigit( split[i] ) && split[i] != '.' && split[i] != '-' )
			{
				opts->label_column = column_num;
				opts->label_defined = true;
				num_labels++;

				// Shortcut out
				is_label = true;
				if ( num_labels > 1 )
				{
					fprintf(
					  stderr,
					  "More than one non-real value in data point: check input file\n" );
					exit( EX_DATAERR );
				}
			}
		}

		// This column is the label column
		if ( opts->label_defined && opts->label_column == column_num )
		{
			if ( is_label )
			{
				data_point.label = malloc( sizeof( char ) * strlen( split ) );
				data_point.label = strdup( split );

				data_point.output_feature = FP_NAN;
				data_point.label_is_double = false;
			}
			else
			{
				data_point.output_feature = strtod( split, NULL );
				data_point.label = NULL;
				data_point.label_is_double = true;
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
			fprintf( stderr, "No label column defined for regression" );
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
	data_pt.output_feature = 42;

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
		printf( "Label: %Lf\nFeatures: ", data_pt.output_feature );
	}
	else
	{
		printf( "Label: %s\nFeatures: ", data_pt.label );
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

distance_t euclid_dist( data_point_t training_pt, data_point_t query_pt )
{
	distance_t distance;
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
	distance.other_data_point = training_pt;

	return distance;
}

void knn( training_data_t data, options_t opts )
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

#ifdef DEBUG
		print_data_point( query_data );
#endif

		knn_distances_t distances;
		distances.num_distances = data.num_samples;
		distances.distances =
		  malloc( sizeof( knn_distances_t ) * data.num_samples );

		for ( int i = 0; i < data.num_samples; ++i )
		{
			distances.distances[i] = euclid_dist( data.samples[i], query_data );
		}

		qsort( distances.distances, distances.num_distances, sizeof( distance_t* ),
		       comp_dist );

		if ( opts.classification )
		{
			int* counts = calloc( opts.k_nearest_neighbors, sizeof( int ) );
			size_t num_uniq_labels = 0;
			char** labels = calloc( opts.k_nearest_neighbors, sizeof( char* ) );
			for ( int i = 0; i < opts.k_nearest_neighbors; ++i )
			{
				for ( int j = 0; j < num_uniq_labels; ++j )
				{
					// Label of current neighbor is in the unique list
					if ( strcmp( labels[j],
					             distances.distances[i].other_data_point->label ) == 0 )
					{
						counts[j]++;
					}
					else
					{
						labels[j] = strdup( distances.distances[i].other_data_point->label );
					}
				}
			}

			for (int i = 0; i < num_uniq_labels; ++i) {
				free(labels[i]);
			}
			free( labels );
			free( counts );
		}
		else
		{
			long double regression_avg = 0;
			for ( int i = 0; i < opts.k_nearest_neighbors; ++i )
			{
				regression_avg +=
				  distances.distances[i].other_data_point->output_feature;
			}
		}

		free( distances.distances );
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
	printf( "Label Column: %zu\n", options.label_column );

#ifdef DEBUG
	for ( int i = 0; i < training_data.num_samples; ++i )
	{
		print_data_point( training_data.samples[i] );
	}
#endif

	knn( training_data, options );

	free_training_data( training_data );

	fclose( data_file );

	return 0;
}
