package tensor

import "fmt"

func Fused0905FloorAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Sigmoid()
	return r, nil
}
func Fused0905FloorAddSigmoidName() string { return "floor_add_sigmoid" }
