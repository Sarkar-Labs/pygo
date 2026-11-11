package tensor

import "fmt"

func Fused0964FloorMatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Floor()
	r = r.Sign()
	return r, nil
}
func Fused0964FloorMatmulSignName() string { return "floor_matmul_sign" }
