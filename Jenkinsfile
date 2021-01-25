pipeline {
    agent {label 'Test'}
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    stages {
        stage ('Build') {
            steps {
                withPythonEnv('python3') {
	                sh 'pip3 install .[hdf5,netcdf] --no-cache-dir'
                    sh 'pip3 install coverage pre-commit --no-cache-dir'
                    sh 'pre-commit run --all-files'
                }
            }
        }
        stage ('Test') {
            steps {
                withPythonEnv('python3') {
                    sh 'COVERAGE_FILE=report/cov/coverage1 mpirun -n 1 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report1.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage2 mpirun -n 2 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report2.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage3 mpirun -n 3 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report3.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage4 mpirun -n 4 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report4.xml heat/'
                }
            }
        }
    }
    post {
        always {
            junit 'report/test/*.xml'
            withPythonEnv('python3') {
                sh 'coverage combine report/cov/*'
                sh 'coverage report'
                sh 'coverage xml'
                sh 'coverage erase report/cov/*'
            }
            withCredentials([string(credentialsId: 'codecov-token', variable: 'CCTOKEN')]) {
                sh 'curl -s https://codecov.io/bash | bash -s -- -c -F unittests -f coverage.xml -t $CCTOKEN  || echo "Codecov failed to upload"'
            }
        }
    }
}
