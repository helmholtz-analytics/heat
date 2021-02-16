pipeline {
    agent {label 'Test'}
    environment {
        PYTHONPATH = "${env.WORKSPACE}"
    }
    stages {
        stage ('Build') {
            steps {
                withPythonEnv('/home/jenkins/allvenvs/') {
	                sh 'pip3 install .[hdf5,netcdf] --no-cache-dir --upgrade --upgrade-strategy eager'
                    sh 'pip3 install pytest coverage pre-commit --no-cache-dir --upgrade'
                    sh 'pre-commit run --all-files'
                }
            }
        }
        stage ('Test') {
            steps {
                withPythonEnv('/home/jenkins/allvenvs/') {
                    sh 'COVERAGE_FILE=report/cov/coverage1 mpirun -n 1 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report1.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage2 mpirun -n 2 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report2.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage3 mpirun -n 3 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report3.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage4 mpirun -n 4 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report4.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage5 mpirun -n 5 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report5.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage6 mpirun -n 6 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report6.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage7 mpirun -n 7 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report7.xml heat/'
                    sh 'COVERAGE_FILE=report/cov/coverage8 mpirun -n 8 coverage run --source=heat --parallel-mode -m pytest --junitxml=report/test/report8.xml heat/'
                }
            }
        }
    }
    post {
        always {
            junit 'report/test/*.xml'
            withPythonEnv('/home/jenkins/allvenvs/') {
                sh 'coverage combine report/cov/*'
                sh 'coverage report'
                sh 'coverage xml'
                sh 'coverage erase report/cov/*'
            }
            withCredentials([string(credentialsId: 'codecov-token', variable: 'CCTOKEN')]) {
                sh 'curl -s https://codecov.io/bash | bash -s -- -c -F unit -f coverage.xml -t $CCTOKEN  || echo "Codecov failed to upload"'
            }
        }
    }
}
