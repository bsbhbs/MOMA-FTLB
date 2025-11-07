package broker;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;


public class MappingBroker extends DatacenterBroker {

    public MappingBroker(String name) throws Exception {
        super(name);
    }

    public void setMapping(int[] mapping) {
        int i = 0;
        for (Cloudlet cl : getCloudletList()) {
            cl.setVmId(mapping[i++]);
        }
    }
    
}
