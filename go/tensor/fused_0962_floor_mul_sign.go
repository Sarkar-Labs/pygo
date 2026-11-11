package tensor

import "fmt"

func Fused0962FloorMulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_sign: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Sign()
	return r, nil
}
func Fused0962FloorMulSignName() string { return "floor_mul_sign" }
