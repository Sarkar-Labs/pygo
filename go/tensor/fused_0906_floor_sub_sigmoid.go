package tensor

import "fmt"

func Fused0906FloorSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Sigmoid()
	return r, nil
}
func Fused0906FloorSubSigmoidName() string { return "floor_sub_sigmoid" }
