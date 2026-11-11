package tensor

import "fmt"

func Fused0075NegAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_add: nil operand")
	}
	r := x.Add(y)
	r = r.Neg()
	return r, nil
}
func Fused0075NegAddName() string { return "neg_add" }
