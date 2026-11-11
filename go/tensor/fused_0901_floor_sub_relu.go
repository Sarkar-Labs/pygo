package tensor

import "fmt"

func Fused0901FloorSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.ReLU()
	return r, nil
}
func Fused0901FloorSubReluName() string { return "floor_sub_relu" }
