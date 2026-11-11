package tensor

import "fmt"

func Fused0937FloorMulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sin: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Sin()
	return r, nil
}
func Fused0937FloorMulSinName() string { return "floor_mul_sin" }
