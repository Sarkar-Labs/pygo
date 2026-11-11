package tensor

import "fmt"

func Fused0922FloorMulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sqrt: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Sqrt()
	return r, nil
}
func Fused0922FloorMulSqrtName() string { return "floor_mul_sqrt" }
