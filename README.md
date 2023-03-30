# Welcome to rosjpt

This package makes joint probability trees (JPTs) available as a ROS service. 
The supported inference types are
- Conditional Inference P(Q|E) (service name: /namespace/infer)
- Most Probable Explanations (MPE) (service name: /namespace/mpe)
- Sampling from the MPE state of a tree (service name: /namespace/sample_mpe)
- applying "permanent" evidence (service name: /namespace/apply_evidence)
- resetting the tree to its original state (service name: /namespace/reset)

The ros interface is described in the `probabilistic_reasoning_msgs` package. 

### Parameters

The only parameter that needs to be set in the ros parameter server is the path of the reasoner. 
The path describes the location of the model on the file system. 

### Launching
Two example launch files are shown in the launch directory.

The ``example.launch`` can be launched if the file `~/Documents/example.jpt` exists by executing
`roslaunch rosjpt example.launch`.

```
<launch>
    <group ns="example_plan">
    <group ns="jpt">
      <param name="path" value="$((env HOME)/Documents/example.jpt" type="string"/>
      <node name="examplejpt" pkg="rosjpt" type="reasoner.py" output="screen"/>
    </group>
  </group>
</launch>
```
While it is not required to provide a namespace it is advisable, since the `path` parameter and service 
name could bite with other ros packages.

### Testing

`rosjpt` comes with a couple of tests. Executing the test suite requires to launch the unittest via roslaunch.
After the launch completed the python unittest framework can be used. 
Open a terminal, build and source your workspace and then execute `roslaunch rosjpt unittesting_tree.launch`.
In a second terminal, run 
```
roscd rosjpt/test/
python test_rosjpt.py 
```

### Example

Example queries for every service are seen in the `test/test_rosjpt.py`. 
In general queries and evidences can be creating through dictionaries mapping from strings to possible values.
For example;
```
mpe = rospy.ServiceProxy("/unittest/jpt/mpe", probabilistic_reasoning_msgs.srv.mpe)
evidence = {"symbolic": "A", "integer": [1, 2, 3], "numeric": [0., 0.5]}
mpe(json.dumps(evidence))
```
asks the model for the most probable explanation, given that the variable named `symbolic` has the value `A`,
the variable named `integer` has the value `1, 2` or `3` and the variable named `numeric` is in the range of 
`0` to `0.5`.

