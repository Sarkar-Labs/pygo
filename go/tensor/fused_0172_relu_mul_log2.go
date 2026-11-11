package tensor

import "fmt"

func Fused0172ReluMulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_mul_log2: nil operand")
	}
	r := x.Mul(y)
	r = r.ReLU()
	r = r.Log2()
	return r, nil
}
func Fused0172ReluMulLog2Name() string { return "relu_mul_log2" }
