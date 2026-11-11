package tensor

import "fmt"

func Fused0902FloorMulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_relu: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.ReLU()
	return r, nil
}
func Fused0902FloorMulReluName() string { return "floor_mul_relu" }
