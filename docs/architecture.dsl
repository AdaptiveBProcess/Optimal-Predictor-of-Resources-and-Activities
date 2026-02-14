workspace {
    !identifiers hierarchical

    model {

        process_analyst = person "Process Analyst" "Designs, runs, and evaluates business process simulations."

        opra_system = softwareSystem "OPRA - Process Simulation & RL Optimization" {
            description "Business process simulation and optimization using reinforcement learning with Top-K / Top-P constrained action spaces."

            /////////////////////////////////////////////////////////////
            // INITIALIZER
            /////////////////////////////////////////////////////////////
            initializer = container "Initializer" {
                description "Bootstraps the simulation, process model, and predictive models."
                technology "Python"
                tags "Initializer"

                load_log = component "Load Event Log" "Loads activities and resources from historical logs."
                create_petri = component "Create Petri Net" "Builds the Petri net representation of the process."
                lift_models = component "Lift Predictive Models" "Configures branching, time, and arrival models."
                trained_decision = component "Models Trained?" "Checks whether trained models exist."
                train_models = component "Train Models" "Trains predictive models from event logs."
                load_models = component "Load Models" "Loads pretrained predictive models."
                calc_defaults = component "Calculate Defaults" "Computes fallback probabilities and durations."
                create_env = component "Create Environment" "Instantiates the simulation environment."
            }

            /////////////////////////////////////////////////////////////
            // ENVIRONMENT
            /////////////////////////////////////////////////////////////
            environment = container "Simulation Environment" {
                description "Discrete-event simulation of the business process."
                technology "Python"
                tags "Environment"

                create_sim = component "Create Simulator" "Initializes the discrete-event simulator."
                init_arrivals = component "Initialize Arrivals" "Generates initial arrival events."
                event_selector = component "Event Selection (FIFO)" "Selects the next enabled event."
                event_prediction = component "Event Prediction Call" "Requests predictions from models."

                // --- Decision Boundary (YELLOW) ---
                calc_available = component "Calculate Available Resources & Top-K / Top-P Activities" "Computes enabled actions and restricts them using Top-K / Top-P." tags "Decision Boundary"
                mask_actions = component "Mask Possible Actions" "Applies feasibility and probability masks."
                execute_action = component "Execute Action" "Executes the selected activity-resource pair."
                calc_reward = component "Calculate Reward" "Computes reward from execution outcome."
                update_env = component "Update Environment" "Advances simulation state."
                terminal_check = component "Terminal?" "Checks episode termination."
                save_log = component "Save Simulation Log" "Persists the simulation event log."
            }

            simulator = container "Discrete-Event Simulator" {
                description "Core simulation engine handling event scheduling and execution."
                technology "Python, SimPy"
                tags "Simulator"

                //



            }

            /////////////////////////////////////////////////////////////
            // AGENT
            /////////////////////////////////////////////////////////////
            agent = container "RL Agent" {
                description "Learns activity-resource allocation policies."
                technology "Python, PyTorch"
                tags "RL Agent"

                start_agent = component "Start Agent" "Initializes the RL agent."
                predict_pair = component "Predict (Activity, Resource)" "Predicts an action from masked space."
                learn = component "Learn" "Updates policy/value networks."
                save_loss = component "Save Loss" "Persists training loss."
            }

            /////////////////////////////////////////////////////////////
            // EVALUATOR
            /////////////////////////////////////////////////////////////
            evaluator = container "Evaluator" {
                description "Evaluates simulation quality and learning behavior."
                technology "Python"
                tags "Evaluator"

                calc_perf = component "Calculate Simulation Performance" "Aggregates KPIs."
                cycle_time = component "Cycle Time" "Computes end-to-end process duration."
                cf_similarity = component "Control-Flow Similarity" "Compares simulated vs real traces."
                loss_evolution = component "Agent Loss Evolution" "Tracks learning stability."
            }

            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – INITIALIZER FLOW
            /////////////////////////////////////////////////////////////
            opra_system.initializer.load_log -> opra_system.initializer.create_petri
            opra_system.initializer.create_petri -> opra_system.initializer.lift_models
            opra_system.initializer.lift_models -> opra_system.initializer.trained_decision

            opra_system.initializer.trained_decision -> opra_system.initializer.train_models "No"
            opra_system.initializer.trained_decision -> opra_system.initializer.load_models "Yes"

            opra_system.initializer.train_models -> opra_system.initializer.calc_defaults
            opra_system.initializer.load_models -> opra_system.initializer.create_env
            opra_system.initializer.calc_defaults -> opra_system.initializer.create_env

            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – ENVIRONMENT LOOP
            /////////////////////////////////////////////////////////////
            opra_system.initializer.create_env -> opra_system.environment.create_sim
            opra_system.environment.create_sim -> opra_system.environment.init_arrivals
            opra_system.environment.init_arrivals -> opra_system.environment.event_selector
            opra_system.environment.event_selector -> opra_system.environment.event_prediction
            opra_system.environment.event_prediction -> opra_system.environment.calc_available
            opra_system.environment.calc_available -> opra_system.environment.mask_actions
            opra_system.environment.mask_actions -> opra_system.environment.execute_action
            opra_system.environment.execute_action -> opra_system.environment.calc_reward
            opra_system.environment.calc_reward -> opra_system.environment.update_env
            opra_system.environment.update_env -> opra_system.environment.terminal_check

            opra_system.environment.terminal_check -> opra_system.environment.save_log "Yes"
            opra_system.environment.terminal_check -> opra_system.environment.event_selector "No"

            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – AGENT INTERACTION
            /////////////////////////////////////////////////////////////
            opra_system.agent.start_agent -> opra_system.agent.predict_pair
            opra_system.agent.predict_pair -> opra_system.environment.calc_available
            opra_system.environment.execute_action -> opra_system.agent.learn
            opra_system.agent.learn -> opra_system.agent.save_loss
            opra_system.agent.save_loss -> opra_system.environment.update_env
            /////////////////////////////////////////////////////////////
            // RELATIONSHIPS – EVALUATION
            /////////////////////////////////////////////////////////////
            opra_system.environment.save_log -> opra_system.evaluator.calc_perf
            opra_system.evaluator.calc_perf -> opra_system.evaluator.cycle_time
            opra_system.evaluator.cycle_time -> opra_system.evaluator.cf_similarity
            opra_system.evaluator.cf_similarity -> opra_system.evaluator.loss_evolution

            /////////////////////////////////////////////////////////////
            // USER
            /////////////////////////////////////////////////////////////
            process_analyst -> opra_system.initializer "Configures & launches"
            process_analyst -> opra_system.evaluator "Analyzes results"
        }
    }

    views {

        container opra_system "OPRA_Containers" {
            include *
            autolayout lr
        }

        component opra_system.initializer "Initializer_Flow" {
            include *
            autolayout lr
        }

        component opra_system.environment "Environment_Flow" {
            include *
            autolayout lr
        }

        component opra_system.agent "Agent_Flow" {
            include *
            autolayout lr
        }

        component opra_system.evaluator "Evaluation_Flow" {
            include *
            autolayout lr
        }

        styles {
            element "Initializer" {
                background #BFE8E3
            }
            element "Environment" {
                background #F28B82
            }
            element "RL Agent" {
                background #8AB4F8
            }
            element "Evaluator" {
                background #E8A8E0
            }
            element "Decision Boundary" {
                background #FDD663
                color #000000
                border dashed
            }
        }
    }
}
