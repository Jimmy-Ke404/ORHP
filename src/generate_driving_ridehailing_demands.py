import os

class split_demands:
    def __init__ (self, file, net_name, penetration_rate=0.5, downtown_facotr=2):
        self.file, self.alpha_RH, self.f_d, self.net_name =\
              file, penetration_rate, downtown_facotr, net_name
        if self.net_name == 'SiouxFalls':
            # Sioux Falls network
            self.downtown_OD = [10, 11, 15, 16, 17, 19]
        elif self.net_name == 'PGH':
            self.downtown_OD = [3093, 2560, 7282, 5513, 7891, 2660]
        else:
            raise ValueError("net_name is not defined properly")
        with open(self.file, "r") as f:
            self.lines = f.readlines()
        self.demand_factor = 2.5 if self.net_name == 'PGH' else 1

    def generate_demand_files(self, folder='splitted_demands'):
        self.alpha_d, self.total_demand = self.cal_alpha_d()
        self.alpha_RH_d = self.f_d * self.alpha_RH / (1 - self.alpha_d + self.alpha_d * self.f_d)
        self.alpha_RH_non_d = self.alpha_RH_d / self.f_d
        self.generate_demands(folder=folder)
        demands_D_file, demands_RH_file = self.generate_demands(folder=folder)

        return demands_D_file, demands_RH_file
    def cal_alpha_d(self):
        downtown_demand = 0
        total_demand = 0
        current_origin = None

        for line in self.lines:
            line = line.rstrip()
            if line.startswith("Origin"):
                origin = int(line.split("Origin")[1])
                current_origin = origin

            elif current_origin != None and len(line) < 3:
                current_origin = None
                
            elif current_origin != None:
                
                to_process = line
                for el in to_process.split(";"):
                    if el == '' or el == ' ':
                        continue;
                    dest = int(el.split(":")[0])
                    demand = float(el.split(":")[1])
                    demand *= self.demand_factor
                    if dest in self.downtown_OD or current_origin in self.downtown_OD:
                        downtown_demand += demand
                    total_demand += demand
                    
        alpha_d = downtown_demand / total_demand

        return alpha_d, total_demand


    def generate_demands(self, folder):
        to_write = [["driving", f"<TOTAL DRVING OD FLOW> \
                     {self.total_demand * (1 - self.alpha_RH)}"], \
                        ["ride hailing", f"<TOTAL RIDE HAILING OD FLOW> \
                     {self.total_demand * self.alpha_RH}"]]
        

        current_origin = None

        for line in self.lines:
            line = line.rstrip()
            if line.startswith("Origin"):
                to_write[0].append(line)
                to_write[1].append(line)
                origin = int(line.split("Origin")[1])
                current_origin = origin

            elif current_origin != None and len(line) < 3:
                to_write[0].append(line)
                to_write[1].append(line)
                current_origin = None
                
            elif current_origin != None:
                line_D = []
                line_RH = []
                for el in line.split(";"):
                    if el == '' or el == ' ':
                        continue;
                    dest = int(el.split(":")[0])
                    demand = float(el.split(":")[1])
                    demand *= self.demand_factor
                    # for OD in downtown, the RH penetration rate is higher
                    if dest in self.downtown_OD or current_origin in self.downtown_OD:
                        demand_RH = demand * self.alpha_RH_d
                        demand_D = demand - demand_RH
                    else:
                        demand_RH = demand * self.alpha_RH_non_d
                        demand_D = demand - demand_RH
                    line_RH.append(":".join((str(dest), str(demand_RH))))
                    line_D.append(":".join((str(dest), str(demand_D))))

                to_write[0].append(";".join(line_D) +';')
                to_write[1].append(";".join(line_RH) + ';')

        # driving demands    
        file_name = f"driving_alphaRH_{self.alpha_RH}_"+ self.file.split('/')[-1]
        path = self.file.split('/')[:-1] + [folder]
        path = os.path.join(*path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, file_name), 'w') as f:
            for line in to_write[0]:
                f.write(line+'\n')
        demands_D_file = os.path.join(path, file_name)

        # ride hailing demands
        file_name = f"ridehailing_alphaRH_{self.alpha_RH}_"+ self.file.split('/')[-1]
        path = self.file.split('/')[:-1] + [folder]
        path = os.path.join(*path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, file_name), 'w') as f:
            for line in to_write[1]:
                f.write(line+'\n')
        demands_RH_file = os.path.join(path, file_name)

        return demands_D_file, demands_RH_file

class generate_pickup_idle_flows:
    def __init__(self, demands_file_D, demands_file_RH, pickup_rate=0.05, idle_rate=0.05) -> None:
        self.D_nodes = set()
        self.demands_D = self.read_demands_from_files(demands_file_D)
        self.demands_RH = self.read_demands_from_files(demands_file_RH)
        self.pickup_rate = pickup_rate
        self.idle_rate = idle_rate

    def add_flows_to_driving_demands(self, file=None):
        self.generate_pickup_flows()
        self.generate_idle_flows()

        with open(file, 'w') as f:
            f.write("driving flows and pickup/idle ridehailing flows \n")
            f.write(f"TOTAL FLOW {self.cal_total_flows(self.demands_D)}\n")
            for origin in self.demands_D:
                f.write(f"Origin: {origin}\n")
                for dest in self.demands_D[origin]:
                    f.write(f"{dest}: {self.demands_D[origin][dest]};\n")
        
    def cal_total_flows(self, demands):
        total_demand = 0
        for o in demands:
            for d in demands[o]:
                demand = demands[o][d]
                total_demand += demand

        return total_demand


    def read_demands_from_files(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        demands = {}

        current_origin = None
        for line in lines:
            line = line.rstrip()
            if line.startswith("Origin"):
                origin = int(line.split("Origin")[1])
                current_origin = origin

            elif current_origin != None and len(line) < 3:
                current_origin = None
                
            elif current_origin != None:
                
                to_process = line
                for el in to_process.split(";"):
                    if el == '' or el == ' ':
                        continue;
                    dest = int(el.split(":")[0])
                    if dest not in self.D_nodes:
                        self.D_nodes.add(dest)
                    demand = float(el.split(":")[1])
                    if current_origin not in demands:
                        demands[current_origin] = {dest: demand}
                    elif dest not in demands[current_origin]:
                        demands[current_origin][dest] = demand
                    else:
                        demands[current_origin][dest] += demand

        return demands


    def generate_pickup_flows(self):
        for origin in self.demands_RH:
            for dest in self.demands_RH[origin]:
                demand_RH = self.demands_RH[origin][dest]
                if origin in self.D_nodes:
                    demand_to_add = demand_RH * self.pickup_rate / (len(self.D_nodes) - 1)
                else:
                    demand_to_add = demand_RH * self.pickup_rate / (len(self.D_nodes))
                for D_node in self.D_nodes:
                    if D_node == origin:
                        continue;
                    # add D_node, origin, demand_to_add to driving demands
                    if D_node in self.demands_D and origin in self.demands_D[D_node]:
                        self.demands_D[D_node][origin] += demand_to_add

    def generate_idle_flows(self):
        for origin in self.demands_RH:
            for dest in self.demands_RH[origin]:
                demand_RH = self.demands_RH[origin][dest]
                idle_flow = demand_RH * self.idle_rate
                self.demands_D[origin][dest] += idle_flow