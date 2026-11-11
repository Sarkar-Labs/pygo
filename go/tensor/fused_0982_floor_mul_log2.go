package tensor

import "fmt"

func Fused0982FloorMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Log2()
	return r, nil
}
func Fused0982FloorMulLog2Name() string { return "floor_mul_log2" }
