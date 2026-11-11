package tensor

import "fmt"

func Fused0967FloorMulGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_gelu: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.GELU()
	return r, nil
}
func Fused0967FloorMulGeluName() string { return "floor_mul_gelu" }
