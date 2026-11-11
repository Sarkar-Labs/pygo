package tensor

import "fmt"

func Fused1487NegMulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_mul_floor: nil operand")
	}
	r := x.Mul(y)
	r = r.Neg()
	r = r.Floor()
	return r, nil
}
func Fused1487NegMulFloorName() string { return "neg_mul_floor" }
